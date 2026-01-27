
Copy of rf-colab-rag-fiqa-tutorial.ipynb
Copy of rf-colab-rag-fiqa-tutorial.ipynb_
  
Join Discord if you need help + ⭐ Star us on GitHub ⭐
👉 Note: This Colab notebook illustrates simplified usage of rapidfireai. For the full RapidFire AI experience with advanced experiment manager, UI, and production features, see our Install and Get Started guide.
🎬 Watch our intro video to get started!
Open In Colab

⚠️ IMPORTANT: Do not let the Colab notebook tab stay idle for more than 5min; Colab will disconnect otherwise. Interact with the cells to avoid disconnection.

Optimizing RAG Pipelines with RapidFire AI
Retrieval-Augmented Generation (RAG) is a practical way to make an AI assistant answer using your documents:

Retrieve: find the most relevant passages for a question.
Generate: give those passages to a language model so it can answer grounded in evidence.
In this beginner-friendly Colab, we’ll build and evaluate a RAG pipeline for a financial opinion Q&A assistant using the FiQA dataset.

Examples of the kind of questions we’re targeting:

“Should I invest in index funds or individual stocks?”
“What’s a good way to save for retirement in my 30s?”
“Is it worth refinancing my mortgage right now?”
What We’re Building
A concrete RAG pipeline that looks like this:

Load a financial corpus (documents + posts).
Split documents into chunks (so we can search smaller, more relevant pieces).
Embed the chunks (turn text into vectors) and store them in a vector index (FAISS).
Retrieve top‑K chunks for each question using similarity search.
(Optional) Rerank the retrieved chunks with a stronger model to keep only the best evidence.
Build a prompt that includes the question + retrieved context.
Generate an answer with a small vLLM model.
Evaluate retrieval quality (Precision, Recall, NDCG@5, MRR) so we can tell which settings find better evidence.
Our Approach
RAG has a lot of “knobs”, and it’s easy to lose track of what helped. In this notebook we’ll focus on retrieval quality by keeping the generator (the vLLM model) fixed and only varying retrieval settings.

We’ll use RapidFireAI to:

Define a small retrieval grid: 2 chunking strategies × 2 reranking top_n values = 4 retrieval configs.
Run all configs the same way on the same dataset.
Compare retrieval metrics side-by-side as they update (Precision/Recall/NDCG/MRR) to pick the best evidence-finding setup.
Install RapidFire AI Package and Setup
Option 1: Install Locally (or on a VM)
For the full RapidFire AI experience—advanced experiment management, UI, and production features—we recommend installing the package on a machine you control (for example, a VM or your local machine) rather than Google Colab. See our Install and Get Started guide.

Option 2: Install in Google Colab
For simplicity, you can run this notebook on Google Colab. This notebook is configured to run end-to-end on Colab with no local installation required.


[14]
0s
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
✅ rapidfireai already installed
✅ langchain_community already installed
Import RapidFire Components
Import RapidFire’s core classes for defining the RAG pipeline and running a small retrieval grid search (plus a Colab-friendly protobuf setting).


[15]
0s
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from rapidfireai import Experiment
from rapidfireai.automl import List, RFLangChainRagSpec, RFvLLMModelConfig, RFPromptManager, RFGridSearch
import re, json
from typing import List as listtype, Dict, Any

# If you get "AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'" from Colab, just rerun this cell
Loading the Data
We load the FiQA queries and relevance labels (qrels), then downsample to keep this Colab run fast. Next we filter the corpus to only documents relevant to the sampled queries and write a smaller corpus_sampled.jsonl. Finally, we update qrels to match the sampled subset so evaluation stays consistent.


[16]
1s
from datasets import load_dataset
import pandas as pd
import json, random
from pathlib import Path

# Dataset directory
dataset_dir = Path("/content/tutorial_notebooks/rag-contexteng/datasets")

# Load full dataset
fiqa_dataset = load_dataset("json", data_files=str(dataset_dir / "fiqa" / "queries.jsonl"), split="train")

Using 6 queries
Found 16 relevant documents for these queries
Sampled 16 documents from 57638 total
Saved to: /content/tutorial_notebooks/rag-contexteng/datasets/fiqa/corpus_sampled.jsonl
Filtered qrels to 16 relevance judgments
Defining the RAG Search Space
This is where RapidFireAI shines. Instead of hardcoding a single RAG configuration, we define a search space using RFLangChainRagSpec.

We will test:

2 Chunking Strategies: Different chunk sizes (256 vs 128).
2 Reranking Strategies: Different top_n values (2 vs 5).
This gives us 8 combinations to evaluate for the retrieval part.


[17]
0s
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

# The EXPERIMENTS list was not used for RFGridSearch and is removed for clarity.
Define Data Processing and Postprocessing Functions
We retrieve context for each question and turn it into LLM-ready prompts. Then we attach the “ground truth” relevant documents from FiQA (qrels) so we can score retrieval quality later.


[18]
0s
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
Define Custom Eval Metrics Functions
The following helper methods compute standard retrieval metrics (Precision, Recall, F1, NDCG@5, MRR) from the retrieved vs. ground-truth document IDs. We compute metrics per batch and then combine them across batches so each config gets one consistent score.


[19]
0s
import math


def compute_ndcg_at_k(retrieved_docs: set, expected_docs: set, k=5):
    """Utility function to compute NDCG@k"""
    relevance = [1 if doc in expected_docs else 0 for doc in list(retrieved_docs)[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

    # IDCG: perfect ranking limited by min(k, len(expected_docs))
    ideal_length = min(k, len(expected_docs))
    ideal_relevance = [3] * ideal_length + [0] * (k - ideal_length)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    return dcg / idcg if idcg > 0 else 0.0


def compute_rr(retrieved_docs: set, expected_docs: set):
    """Utility function to compute Reciprocal Rank (RR) for a single query"""
    rr = 0
    for i, retrieved_doc in enumerate(retrieved_docs):
        if retrieved_doc in expected_docs:
            rr = 1 / (i + 1)
            break
    return rr


def sample_compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    """Function to compute all eval metrics based on retrievals and/or generations"""

    true_positives, precisions, recalls, f1_scores, ndcgs, rrs = 0, [], [], [], [], []
    total_queries = len(batch["query"])

    for pred, gt in zip(batch["retrieved_documents"], batch["ground_truth_documents"]):
        expected_set = set(gt)
        retrieved_set = set(pred)

        true_positives = len(expected_set.intersection(retrieved_set))
        precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0
        recall = true_positives / len(expected_set) if len(expected_set) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        ndcgs.append(compute_ndcg_at_k(retrieved_set, expected_set, k=5))
        rrs.append(compute_rr(retrieved_set, expected_set))

    return {
        "Total": {"value": total_queries},
        "Precision": {"value": sum(precisions) / total_queries},
        "Recall": {"value": sum(recalls) / total_queries},
        "F1 Score": {"value": sum(f1_scores) / total_queries},
        "NDCG@5": {"value": sum(ndcgs) / total_queries},
        "MRR": {"value": sum(rrs) / total_queries},
    }


def sample_accumulate_metrics_fn(
    aggregated_metrics: Dict[str, listtype],
) -> Dict[str, Dict[str, Any]]:
    """Function to accumulate eval metrics across all batches"""

    num_queries_per_batch = [m["value"] for m in aggregated_metrics["Total"]]
    total_queries = sum(num_queries_per_batch)
    algebraic_metrics = ["Precision", "Recall", "F1 Score", "NDCG@5", "MRR"]

    return {
        "Total": {"value": total_queries},
        **{
            metric: {
                "value": sum(
                    m["value"] * queries
                    for m, queries in zip(
                        aggregated_metrics[metric], num_queries_per_batch
                    )
                )
                / total_queries,
                "is_algebraic": True,
                "value_range": (0, 1),
            }
            for metric in algebraic_metrics
        },
    }
Define Partial Multi-Config Knobs for vLLM Generator part of RAG Pipeline using RapidFire AI Wrapper APIs
We pick a lightweight vLLM model and sampling settings that fit in Colab GPU memory. Then we bundle the generator + our preprocessing/metrics functions into config_set, which RapidFire will run across the 4 retrieval configs.


[20]
0s
import copy

# The base vLLM config (without rag spec, as rag will be varied in the loop)
base_vllm_config_params = {
    "model_config": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": "half",
        "gpu_memory_utilization": 0.25,
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "max_model_len": 3000,
        "disable_log_stats": True,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
    },
    "sampling_params": {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 128,
    },
    "prompt_manager": None,
}

# Extracting the variable parts from the 'rag_gpu' object defined previously
chunk_splitter_definitions = rag_gpu.text_splitter.list_values if hasattr(rag_gpu.text_splitter, 'list_values') else [rag_gpu.text_splitter]
reranker_top_ns = rag_gpu.reranker_kwargs["top_n"].list_values if hasattr(rag_gpu.reranker_kwargs["top_n"], 'list_values') else [rag_gpu.reranker_kwargs["top_n"]]

# Prepare a list of RFvLLMModelConfig objects for RFGridSearch
list_of_vllm_configs = []

for splitter_instance in chunk_splitter_definitions:
    for top_n_val in reranker_top_ns:
        # Create a deep copy of the base rag_gpu object to modify it for the current iteration
        current_rag_spec = copy.deepcopy(rag_gpu)

        # Set the specific text_splitter for this iteration
        current_rag_spec.text_splitter = splitter_instance

        # Set the specific top_n for this iteration within the reranker_kwargs
        current_rag_spec.reranker_kwargs["top_n"] = top_n_val

        # Create the RFvLLMModelConfig for this combination
        current_vllm_config = RFvLLMModelConfig(
            rag=current_rag_spec,
            **base_vllm_config_params
        )
        list_of_vllm_configs.append(current_vllm_config)

batch_size = 3 # Smaller batch size for generation
config_set = {
    # The 'vllm_config' key now holds a List of RFvLLMModelConfig objects,
    # which RFGridSearch will iterate over.
    "vllm_config": List(list_of_vllm_configs),
    "batch_size": batch_size,
    "preprocess_fn": sample_preprocess_fn,
    "postprocess_fn": sample_postprocess_fn,
    "compute_metrics_fn": sample_compute_metrics_fn,
    "accumulate_metrics_fn": sample_accumulate_metrics_fn,
    "online_strategy_kwargs": {
        "strategy_name": "normal",
        "confidence_level": 0.95,
        "use_fpc": True,
    },
}
Create Config Group
We create an RFGridSearch over config_set, producing 4 retrieval configs (2 chunkers × 2 rerankers) to run and compare.


[21]
0s
# Simple grid search across all config combinations: 4 total (2 chunkers × 2 rerankers)
config_group = RFGridSearch(config_set)
Create Experiment
List item
List item
An Experiment is RapidFire’s top-level container for this notebook run: it groups configs/runs, saves artifacts, and tracks metrics under a unique name. We set mode="evals" because we’re running evaluation (not training). See the docs: https://oss-docs.rapidfire.ai/en/latest/experiment.html#api-experiment


[12]
50s
experiment = Experiment(experiment_name="exp1-fiqa-rag-colab", mode="evals")

Display Ray Dashboard (Optional)
Ray is the system RapidFire uses under the hood to run work in parallel; this cell simply embeds Ray’s dashboard below so we can monitor what’s running.


[13]
0s
# Display the Ray dashboard in the Colab notebook
from google.colab import output
output.serve_kernel_port_as_iframe(8855)

Run Multi-Config Evals + Launch Interactive Run Controller
Now we get to the main function for running multi-config evals. Two tables will appear below the run_evals cell:

The first table will appear immediately. It lists all preprocessing/RAG sources.
After a short while, the second table will appear. It lists all individual runs with their knobs and metrics that are updated in real-time via online aggregation showing both estimates and confidence intervals.
RapidFire AI also provides an Interactive Controller panel UI for Colab that lets you manage executing runs dynamically in real-time from the notebook:

⏹️ Stop: Gracefully stop a running config
▶️ Resume: Resume a stopped run
🗑️ Delete: Remove a run from this experiment
📋 Clone: Create a new run by editing the config dictionary of a parent run to try new knob values; optional warm start of parameters
🔄 Refresh: Update run status and metrics

[22]
10m
# Launch evals of all RAG configs in the config_group with swap granularity of 4 chunks
# NB: If your machine has more than 1 GPU, set num_actors to that number
results = experiment.run_evals(
    config_group=config_group,
    dataset=fiqa_dataset,
    num_actors=1,
    num_shards=4,
    seed=42,
)

View Results

[23]
0s
# Convert results dict to DataFrame
results_df = pd.DataFrame([
    {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in {**metrics_dict, 'run_id': run_id}.items()}
    for run_id, (_, metrics_dict) in results.items()
])

results_df

Next steps:
End Experiment

[24]
from google.colab import output
from IPython.display import display, HTML

display(HTML('''
<button id="continue-btn" style="padding: 10px 20px; font-size: 16px;">Click to End Experiment</button>
'''))

# eval_js blocks until the Promise resolves
output.eval_js('''
new Promise((resolve) => {
    document.getElementById("continue-btn").onclick = () => {
        document.getElementById("continue-btn").disabled = true;
        document.getElementById("continue-btn").innerText = "Continuing...";
        resolve("clicked");
    };
})
''')

# Actually end the experiment after the button is clicked
experiment.end()
print("Done!")

View RapidFire AI Log Files

[ ]
# Get the experiment-specific log file
log_file = experiment.get_log_file_path()

print(f"📄 Log File: {log_file}")
print()

if log_file.exists():
    print("=" * 80)
    print(f"Last 30 lines of {log_file.name}:")
    print("=" * 80)
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[-30:]:
            print(line.rstrip())
else:
    print(f"❌ Log file not found: {log_file}")
Conclusion
We built a simple Financial Q&A RAG pipeline and compared 4 retrieval configurations (chunking × reranking) using standard retrieval metrics.

Optional ideas to explore later:

Increase sample_fraction (or run locally) for more reliable results.
Try additional retrieval knobs (e.g., embedding model, k, chunk overlap) and re-run the same evaluation loop.
Colab paid products - Cancel contracts here
