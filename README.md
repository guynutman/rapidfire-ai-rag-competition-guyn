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

---

## ⚠️ Submission Note

This submission was completed under significant time pressure near the competition deadline. While the experiment ran successfully and produced valid results, some documentation aspects are incomplete:

- Full analysis of all configurations is in progress
- Additional screenshots and detailed metrics visualization pending
- Complete knob extraction and justification to be added post-submission

The core experiment, results CSV, and key findings are included. I plan to update this repository with complete documentation after the deadline if permitted.

---
