import pytest

from neuroflow.fusion_rrf import convex_combination_fusion, reciprocal_rank_fusion


def test_rrf_scoring_math():
    # doc1 is rank 1 in both BM25 and Dense
    # score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
    bm25 = [({"chunk_id": "doc1", "text": "a"}, 10.0), ({"chunk_id": "doc2", "text": "b"}, 5.0)]
    dense = [({"chunk_id": "doc1", "text": "a"}, 0.95), ({"chunk_id": "doc3", "text": "c"}, 0.85)]

    fused = reciprocal_rank_fusion(bm25, dense, k=60)
    assert len(fused) == 3
    assert fused[0][0]["chunk_id"] == "doc1"
    assert pytest.approx(fused[0][1], 0.0001) == (1.0 / 61.0 + 1.0 / 61.0)


def test_convex_combination_fusion_math():
    bm25 = [({"chunk_id": "doc1", "text": "a"}, 10.0), ({"chunk_id": "doc2", "text": "b"}, 0.0)]
    dense = [({"chunk_id": "doc1", "text": "a"}, 0.5), ({"chunk_id": "doc2", "text": "b"}, 1.0)]

    # Alpha = 0.5 -> average of normalized scores
    fused = convex_combination_fusion(bm25, dense, alpha=0.5)
    assert len(fused) == 2
    # doc1: bm25_norm=1.0, dense_norm=0.0 -> score = 0.5*0.0 + 0.5*1.0 = 0.5
    # doc2: bm25_norm=0.0, dense_norm=1.0 -> score = 0.5*1.0 + 0.5*0.0 = 0.5
    assert pytest.approx(fused[0][1], 0.001) == 0.5
