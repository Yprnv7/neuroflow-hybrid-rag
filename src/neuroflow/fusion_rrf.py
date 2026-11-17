from typing import Any, Dict, List, Tuple


def reciprocal_rank_fusion(
    bm25_results: List[Tuple[Dict[str, Any], float]],
    dense_results: List[Tuple[Dict[str, Any], float]],
    k: int = 60
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Reciprocal Rank Fusion (RRF):
    score(d) = sum_{system} [ 1.0 / (k + rank_{system}(d)) ]
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
