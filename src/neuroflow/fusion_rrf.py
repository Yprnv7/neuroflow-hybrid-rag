"""Reciprocal Rank Fusion (RRF) and Convex Score Combination.

RRF provides rank-based, parameter-free candidate merging robust to disparate score distributions.
Convex combination provides score-magnitude fusion via an alpha parameter [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Dict[str, Any], float]],
    dense_results: List[Tuple[Dict[str, Any], float]],
    k: int = 60,
) -> List[Tuple[Dict[str, Any], float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF(d) = sum_{system} [ 1.0 / (k + rank_{system}(d)) ]
    where rank is 1-based index in each retriever's ranked output.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))

    for rank, (chunk, _) in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in ranked]


def _normalize_scores(results: List[Tuple[Dict[str, Any], float]]) -> Dict[str, float]:
    """Min-max normalize raw scores to [0.0, 1.0]."""
    if not results:
        return {}
    raw_scores = [s for _, s in results]
    min_s = min(raw_scores)
    max_s = max(raw_scores)
    diff = max_s - min_s
    if diff <= 1e-9:
        return {c["chunk_id"]: 1.0 for c, _ in results}
    return {c["chunk_id"]: (s - min_s) / diff for c, s in results}


def convex_combination_fusion(
    bm25_results: List[Tuple[Dict[str, Any], float]],
    dense_results: List[Tuple[Dict[str, Any], float]],
    alpha: float = 0.5,
) -> List[Tuple[Dict[str, Any], float]]:
    """Convex combination of normalized retrieval scores:

    score(d) = alpha * score_dense(d) + (1.0 - alpha) * score_bm25(d)

    alpha = 1.0 -> purely dense
    alpha = 0.0 -> purely BM25
    alpha = 0.5 -> equal contribution
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    bm25_norm = _normalize_scores(bm25_results)
    dense_norm = _normalize_scores(dense_results)

    all_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for c, _ in bm25_results:
        chunk_map[c["chunk_id"]] = c
    for c, _ in dense_results:
        chunk_map[c["chunk_id"]] = c

    fused_scores: Dict[str, float] = {}
    for cid in all_ids:
        s_sparse = bm25_norm.get(cid, 0.0)
        s_dense = dense_norm.get(cid, 0.0)
        fused_scores[cid] = alpha * s_dense + (1.0 - alpha) * s_sparse

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in ranked]
