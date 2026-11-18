# Retrieval Evaluation Audit: SciFact Benchmark & Latency Analysis

**Date:** 2026-11-18  
**Author:** Pranav Yeleswarapu  
**Repository:** `neuroflow-hybrid-rag`  

---

## 1. Executive Summary

A critical audit of `scripts/evaluate_beir.py` and the initial benchmark report in `docs/BENCHMARKS.md` revealed that the reported metrics—specifically **NDCG@10 = 1.0000**, **Recall@10 = 1.0000**, and **$p95$ Latency = 0.25 ms**—were artifacts of a synthetic testing harness rather than genuine empirical evaluation on the BEIR SciFact benchmark. 

In zero-shot information retrieval, perfect scores ($1.0000$) do not occur on standard research benchmarks. Published SOTA retrievers on SciFact score between **$0.66$ and $0.78$ NDCG@10**. Furthermore, running a 6-layer Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) over 25 candidate passages on CPU takes **$15$ to $40\text{ ms}$**, making a $0.25\text{ ms}$ tail latency physically impossible for neural reranking.

This document records the exact root causes discovered in the codebase and specifies the rigorous remediation applied.

---

## 2. Root Cause Analysis

Inspection of `scripts/evaluate_beir.py` identified four fatal flaws in the original benchmark harness:

### 2.1 Synthetic Micro-Corpus Instead of Official BEIR Split
```python
# scripts/evaluate_beir.py (legacy lines 11-25)
CORPUS = [
    {"doc_id": "sci-01", "text": "CRISPR-Cas9 enables targeted genome editing..."},
    {"doc_id": "sci-02", "text": "Telomerase reverse transcriptase maintains..."},
    {"doc_id": "sci-03", "text": "Mitochondrial oxidative phosphorylation..."},
    {"doc_id": "sci-04", "text": "Transformer neural networks use multi-head..."},
    {"doc_id": "sci-05", "text": "Distributed consensus algorithms ensure..."},
]

QUERIES = [
    ("How does CRISPR-Cas9 edit genomes?", "sci-01"),
    ("What role does telomerase play in chromosomes?", "sci-02"),
    ("Explain ATP generation in mitochondria", "sci-03"),
    ("How do transformer attention mechanisms work?", "sci-04"),
    ("What guarantees state machine safety in distributed systems?", "sci-05"),
]
```
* **Finding:** The evaluation was executed against **5 hand-crafted documents** and **5 synthetic queries** constructed to match the vocabulary of the documents directly.
* **Impact:** The official BEIR SciFact dataset contains **5,183 scientific abstracts**, **300 test queries**, and **339 expert relevance judgments (qrels)**. Searching 5 mutually exclusive domain documents trivially produces $100\%$ precision for any basic keyword search.

### 2.2 Forced Metric Assignment in Benchmark Script
```python
# scripts/evaluate_beir.py (legacy line 91)
recall = metrics[mode]["hits"] / len(QUERIES)
ndcg = 1.0 if recall == 1.0 else round(recall * 0.92, 4)
```
* **Finding:** NDCG was never calculated using standard Discounted Cumulative Gain formulas ($DCG_p = \sum_{i=1}^p \frac{2^{rel_i} - 1}{\log_2(i + 1)}$). Instead, a hardcoded conditional forced `ndcg = 1.0` whenever `recall == 1.0`.

### 2.3 Fallback Encoders Bypassing Neural Inference
```python
# scripts/evaluate_beir.py (legacy lines 38, 41)
encoder = DenseEncoder(fallback=True)
reranker = CrossEncoderReranker(fallback=True)
```
* **Finding:** The benchmark instantiated encoders with `fallback=True`. This bypassed PyTorch and Hugging Face transformer models entirely, replacing embedding projection with hash-based bag-of-words vectors and reranking with simple token overlap counts.
* **Impact:** The measured $0.25\text{ ms}$ latency was simply the time required for Python to compute dot products between five 64-dimensional float arrays in memory. Real transformer inference was never executed.

---

## 3. Academic & Industry Baselines for Context

To establish realistic expectations for SciFact evaluation, below are published benchmark figures from the original BEIR paper (*Thakur et al., NeurIPS 2021*) and the Massive Text Embedding Benchmark (MTEB):

| Model / Architecture | NDCG@10 | Recall@100 | Measured CPU Latency (ms) | Source |
| :--- | :--- | :--- | :--- | :--- |
| **BM25 (Anserini / Okapi)** | **0.665** | 0.852 | ~2–5 ms | Thakur et al. (2021) |
| **Contriever (Unsupervised Dense)** | **0.677** | 0.901 | ~8–15 ms (encode + vector search) | Izacard et al. (2022) |
| **bge-small-en-v1.5 (Dense)** | **0.723** | 0.914 | ~10–20 ms (CPU) | MTEB Leaderboard (2024) |
| **BM25 + Dense (RRF $k=60$)** | **0.738** | 0.932 | ~12–25 ms | Empirical Hybrid RAG |
| **Hybrid + MiniLM Cross-Encoder** | **0.756** | 0.932 | **35–65 ms** (25 candidates on CPU) | BEIR Reranking Standard |

---

## 4. Remediation Plan Executed

1. **Official Dataset Integration:** Download and cache the official BEIR SciFact dataset (5,183 documents, 300 test queries, 339 qrels).
2. **Canonical Metric Computation:** Use `ranx` and standard TREC evaluation formulas (`NDCG@10`, `Recall@100`, `MRR@10`) over the full candidate pool.
3. **Real Transformer Models:**
   - Dense: `BAAI/bge-small-en-v1.5` (384-dimensional embeddings).
   - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (evaluating top-25 and top-50 candidate pools).
4. **Honest Latency Decomposition:** Benchmark each pipeline stage separately (Query Embed, BM25 Index Search, Dense Vector Search, Fusion, Cross-Encoder Rerank) across 300 test queries on Apple Silicon CPU.
5. **Multi-Dataset Verification:** Evaluate on a second benchmark dataset (NFCorpus or FiQA) to demonstrate generalizability beyond a single scientific domain.
