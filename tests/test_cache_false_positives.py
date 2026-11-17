import numpy as np

from neuroflow.semantic_cache import SemanticCache


def test_semantic_cache_isolation_and_threshold():
    cache = SemanticCache(similarity_threshold=0.92)

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # v2 is similar (cos_sim ~ 0.99)
    v2 = np.array([0.99, 0.05, 0.0], dtype=np.float32)
    v2 = v2 / np.linalg.norm(v2)

    # v3 is orthogonal
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    cache.put(tenant_id="tenant_a", query="q1", query_vec=v1, response={"ans": "A"})

    # Hit for tenant_a with v2
    res_a = cache.get(tenant_id="tenant_a", query_vec=v2)
    assert res_a is not None
    assert res_a[0]["ans"] == "A"

    # Miss for tenant_b with identical vector (Strict multi-tenant isolation)
    res_b = cache.get(tenant_id="tenant_b", query_vec=v2)
    assert res_b is None

    # Miss for tenant_a with v3 (dissimilar query)
    res_a_miss = cache.get(tenant_id="tenant_a", query_vec=v3)
    assert res_a_miss is None
