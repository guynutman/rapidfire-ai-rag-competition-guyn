# RapidFire AI RAG Experiment – Winter Competition

This repo contains a retrieval-first RAG experimentation notebook using RapidFire AI.

## Use case
Answer financial QA queries using a small document corpus, optimizing chunking and reranking.

## Experiments
We vary:
- Chunk size (256 vs 128)
- Reranker top-n (2 vs 5)

All experiments are run using RapidFire AI’s hyperparallel evaluation API.

## Artifacts
- Google Colab notebook
- Retrieval metrics plots
- RapidFire logs and screenshots

## Notes
This project focuses on retrieval quality (Recall@K, MRR) per RAG Track guidelines.

