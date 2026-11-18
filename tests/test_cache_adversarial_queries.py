"""Tests for multi-tenant semantic cache isolation, LFU eviction, and near-duplicate adversarial queries."""

import numpy as np

from neuroflow.dense_encoder import DenseEncoder
from neuroflow.semantic_cache import SemanticCache


def test_strict_multi_tenant_isolation():
    """Verify Tenant A's cached response is NEVER returned to Tenant B, even on identical query."""
    cache = SemanticCache(similarity_threshold=0.90)
    encoder = DenseEncoder(fallback=True)

    query = "What is the company severance policy?"
    q_vec = encoder.encode_one(query)

    cache.put(tenant_id="tenant_alpha", query=query, query_vec=q_vec, response={"answer": "Alpha Confidential"})

    # Tenant Alpha should get hit
    hit_alpha = cache.get(tenant_id="tenant_alpha", query_vec=q_vec)
    assert hit_alpha is not None
    assert hit_alpha[0]["answer"] == "Alpha Confidential"

    # Tenant Beta with identical vector MUST get a cache miss
    hit_beta = cache.get(tenant_id="tenant_beta", query_vec=q_vec)
    assert hit_beta is None


def test_tenant_erasure_purges_cache():
    """Verify GDPR forget_tenant removes all entries for specified tenant."""
    cache = SemanticCache(similarity_threshold=0.90)
    encoder = DenseEncoder(fallback=True)

    vec = encoder.encode_one("quarterly tax filing")
    cache.put(tenant_id="tenant_erase", query="quarterly tax filing", query_vec=vec, response={"tax": 100})
    assert cache.get(tenant_id="tenant_erase", query_vec=vec) is not None

    erased_count = cache.forget_tenant("tenant_erase")
    assert erased_count == 1
    assert cache.get(tenant_id="tenant_erase", query_vec=vec) is None


def test_lfu_eviction_under_capacity_limit():
    """Verify cache evicts Least Frequently Used (LFU) entries when max capacity is hit."""
    cache = SemanticCache(similarity_threshold=0.85, max_entries_per_tenant=3)

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v4 = np.array([0.7, 0.7, 0.0], dtype=np.float32)

    cache.put("t1", "q1", v1, {"id": 1})
    cache.put("t1", "q2", v2, {"id": 2})
    cache.put("t1", "q3", v3, {"id": 3})

    # Access q1 and q2 multiple times to raise their hit count
    cache.get("t1", v1)
    cache.get("t1", v1)
    cache.get("t1", v2)

    # Insert 4th entry -> q3 (0 hits) should be evicted
    cache.put("t1", "q4", v4, {"id": 4})

    assert len(cache.cache["t1"]) == 3
    remaining_queries = [e["query"] for e in cache.cache["t1"]]
    assert "q3" not in remaining_queries
    assert "q1" in remaining_queries
    assert "q4" in remaining_queries
