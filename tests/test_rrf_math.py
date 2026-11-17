import pytest

from neuroflow.fusion_rrf import reciprocal_rank_fusion


def test_rrf_scoring_math():
    # doc1 is rank 1 in both BM25 and Dense
    # score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
    bm25 = [({"chunk_id": "doc1", "text": "a"}, 10.0), ({"chunk_id": "doc2", "text": "b"}, 5.0)]
    dense = [({"chunk_id": "doc1", "text": "a"}, 0.95), ({"chunk_id": "doc3", "text": "c"}, 0.85)]

    fused = reciprocal_rank_fusion(bm25, dense, k=60)
    assert len(fused) == 3
    assert fused[0][0]["chunk_id"] == "doc1"
    assert pytest.approx(fused[0][1], 0.0001) == (1.0 / 61.0 + 1.0 / 61.0)
