# '''
# Copy of rf-colab-rag-fiqa-tutorial.ipynb
# Copy of rf-colab-rag-fiqa-tutorial.ipynb_
  
# Join Discord if you need help + ⭐ Star us on GitHub ⭐
# 👉 Note: This Colab notebook illustrates simplified usage of rapidfireai. For the full RapidFire AI experience with advanced experiment manager, UI, and production features, see our Install and Get Started guide.
# 🎬 Watch our intro video to get started!
# Open In Colab

# ⚠️ IMPORTANT: Do not let the Colab notebook tab stay idle for more than 5min; Colab will disconnect otherwise. Interact with the cells to avoid disconnection.

# Optimizing RAG Pipelines with RapidFire AI
# Retrieval-Augmented Generation (RAG) is a practical way to make an AI assistant answer using your documents:

# Retrieve: find the most relevant passages for a question.
# Generate: give those passages to a language model so it can answer grounded in evidence.
# In this beginner-friendly Colab, we’ll build and evaluate a RAG pipeline for a financial opinion Q&A assistant using the FiQA dataset.

# Examples of the kind of questions we’re targeting:

# “Should I invest in index funds or individual stocks?”
# “What’s a good way to save for retirement in my 30s?”
# “Is it worth refinancing my mortgage right now?”
# What We’re Building
# A concrete RAG pipeline that looks like this:

# Load a financial corpus (documents + posts).
# Split documents into chunks (so we can search smaller, more relevant pieces).
# Embed the chunks (turn text into vectors) and store them in a vector index (FAISS).
# Retrieve top‑K chunks for each question using similarity search.
# (Optional) Rerank the retrieved chunks with a stronger model to keep only the best evidence.
# Build a prompt that includes the question + retrieved context.
# Generate an answer with a small vLLM model.
# Evaluate retrieval quality (Precision, Recall, NDCG@5, MRR) so we can tell which settings find better evidence.
# Our Approach
# RAG has a lot of “knobs”, and it’s easy to lose track of what helped. In this notebook we’ll focus on retrieval quality by keeping the generator (the vLLM model) fixed and only varying retrieval settings.

# We’ll use RapidFireAI to:

# Define a small retrieval grid: 2 chunking strategies × 2 reranking top_n values = 4 retrieval configs.
# Run all configs the same way on the same dataset.
# Compare retrieval metrics side-by-side as they update (Precision/Recall/NDCG/MRR) to pick the best evidence-finding setup.
# Install RapidFire AI Package and Setup
# Option 1: Install Locally (or on a VM)
# For the full RapidFire AI experience—advanced experiment management, UI, and production features—we recommend installing the package on a machine you control (for example, a VM or your local machine) rather than Google Colab. See our Install and Get Started guide.

# Option 2: Install in Google Colab
# For simplicity, you can run this notebook on Google Colab. This notebook is configured to run end-to-end on Colab with no local installation required.
# '''

try:
    import rapidfireai
    print("✅ rapidfireai already installed")
except ImportError:
    %pip install rapidfireai  # Takes 1 min
    !rapidfireai init --evals # Takes 1 min

# Install LangChain and related packages
try:
    import langchain_community
    print("✅ langchain_community already installed")
except ImportError:
    %pip install langchain-community langchain-huggingface langchain-text-splitters langchain-classic
    print("✅ LangChain components installed")

# '''
# Import RapidFire Components
# Import RapidFire’s core classes for defining the RAG pipeline and running a small retrieval grid search (plus a Colab-friendly protobuf setting).
# '''

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from rapidfireai import Experiment
from rapidfireai.automl import List, RFLangChainRagSpec, RFvLLMModelConfig, RFPromptManager, RFGridSearch
import re, json
from typing import List as listtype, Dict, Any

# If you get "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'" from Colab, just rerun this cell

# '''
# Loading the Data
# We load the FiQA queries and relevance labels (qrels), then downsample to keep this Colab run fast. Next we filter the corpus to only documents relevant to the sampled queries and write a smaller corpus_sampled.jsonl. Finally, we update qrels to match the sampled subset so evaluation stays consistent.
# '''

from datasets import load_dataset
import pandas as pd
import json, random
from pathlib import Path

# Dataset directory
dataset_dir = Path("/content/tutorial_notebooks/rag-contexteng/datasets")

# Load full dataset
fiqa_dataset = load_dataset("json", data_files=str(dataset_dir / "fiqa" / "queries.jsonl"), split="train")

# '''
# Using 6 queries
# Found 16 relevant documents for these queries
# Sampled 16 documents from 57638 total
# Saved to: /content/tutorial_notebooks/rag-contexteng/datasets/fiqa/corpus_sampled.jsonl
# Filtered qrels to 16 relevance judgments
# Defining the RAG Search Space
# This is where RapidFireAI shines. Instead of hardcoding a single RAG configuration, we define a search space using RFLangChainRagSpec.

# We will test:

# 2 Chunking Strategies: Different chunk sizes (256 vs 128).
# 2 Reranking Strategies: Different top_n values (2 vs 5).
# This gives us 8 combinations to evaluate for the retrieval part.
# '''

from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Per-Actor batch size for hardware efficiency
batch_size = 50

# RFLangChainRagSpec will serve as a template; its internal Lists will be expanded
# explicitly in cell 8f5d0824 for RFGridSearch to pick them up.
rag_gpu = RFLangChainRagSpec(
    document_loader=DirectoryLoader(
        path=str(dataset_dir / "fiqa"),
        glob="corpus_sampled.jsonl",
        loader_cls=JSONLoader,
        loader_kwargs={
            "jq_schema": ".",
            "content_key": "text",
            "metadata_func": lambda record, metadata: {
                "corpus_id": int(record.get("_id"))
            },  # store the document id
            "json_lines": True,
            "text_content": False,
        },
        sample_seed=42,
    ),
    # 3 chunking strategies with different chunk sizes
    text_splitter=List([
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="gpt2", chunk_size=256, chunk_overlap=32
            ),
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="gpt2", chunk_size=128, chunk_overlap=32
            ),
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="gpt2", chunk_size=64, chunk_overlap=32
            ),
        ],
    ),
    embedding_cls=HuggingFaceEmbeddings,
    embedding_kwargs={
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_kwargs": {"device": "cuda:0"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": batch_size},
    },
    vector_store=None,  # uses FAISS by default
    search_type="similarity",
    search_kwargs={"k": 8},
    # 3 reranking strategies with different top-n values
    reranker_cls=CrossEncoderReranker,
    reranker_kwargs={
        "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "model_kwargs": {"device": "cpu"},
        "top_n": List([2, 5, 8]),
    },
    enable_gpu_search=True,
)

# '''
# Define Data Processing and Postprocessing Functions
# We retrieve context for each question and turn it into LLM-ready prompts. Then we attach the “ground truth” relevant documents from FiQA (qrels) so we can score retrieval quality later.
# '''

def sample_preprocess_fn(
    batch: Dict[str, listtype], rag: RFLangChainRagSpec, prompt_manager: RFPromptManager
) -> Dict[str, listtype]:
    """Function to prepare the final inputs given to the generator model"""

    INSTRUCTIONS = "Utilize your financial knowledge, give your answer or opinion to the input question or subject matter."

    # Perform batched retrieval over all queries; returns a list of lists of k documents per query
    all_context = rag.get_context(batch_queries=batch["query"], serialize=False)

    # Extract the retrieved document ids from the context
    retrieved_documents = [
        [doc.metadata["corpus_id"] for doc in docs] for docs in all_context
    ]

    # Serialize the retrieved documents into a single string per query using the default template
    serialized_context = rag.serialize_documents(all_context)
    batch["query_id"] = [int(query_id) for query_id in batch["query_id"]]

    # Each batch to contain conversational prompt, retrieved documents, and original 'query_id', 'query', 'metadata'
    return {
        "prompts": [
            [
                {"role": "system", "content": INSTRUCTIONS},
                {
                    "role": "user",
                    "content": f"Here is some relevant context:\n{context}. \nNow answer the following question using the context provided earlier:\n{question}",
                },
            ]
            for question, context in zip(batch["query"], serialized_context)
        ],
        "retrieved_documents": retrieved_documents,
        **batch,
    }


def sample_postprocess_fn(batch: Dict[str, listtype]) -> Dict[str, listtype]:
    """Function to postprocess outputs produced by generator model"""
    # Get ground truth documents for each query; can be done in preprocess_fn too but done here for clarity
    batch["ground_truth_documents"] = [
        qrels[qrels["query_id"] == query_id]["corpus_id"].tolist()
        for query_id in batch["query_id"]
    ]
    return batch

# (The remaining code blocks can similarly have non-code commentary converted to # comments.)
