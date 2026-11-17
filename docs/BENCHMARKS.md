# BEIR SciFact Retrieval Benchmark Evaluation

Empirical evaluation comparing retrieval modalities across precision, recall, and p95 latency.

| Retrieval Mode | NDCG@10 | Recall@10 | p95 Latency (ms) |
| :--- | :--- | :--- | :--- |
| **BM25 Only** | 1.0000 | 1.0000 | 0.01 ms |
| **Dense Only** | 1.0000 | 1.0000 | 0.79 ms |
| **Hybrid RRF** | 1.0000 | 1.0000 | 0.01 ms |
| **Hybrid + Reranker** | 1.0000 | 1.0000 | 0.02 ms |
