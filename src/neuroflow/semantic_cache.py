import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SemanticCache:
    """
    Tenant-isolated semantic cache with similarity calibration.
    Guarantees strict GDPR isolation between distinct tenant IDs.
    """

    def __init__(self, similarity_threshold: float = 0.92, max_entries_per_tenant: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_entries_per_tenant = max_entries_per_tenant
        # tenant_id -> list of cache entries
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, tenant_id: str, query_vec: np.ndarray) -> Optional[Tuple[Dict[str, Any], float]]:
        entries = self.cache.get(tenant_id, [])
        if not entries:
            self.stats["misses"] += 1
            return None

        matrix = np.vstack([e["vector"] for e in entries])
        qn = np.linalg.norm(query_vec)
        if qn > 0:
            query_vec = query_vec / qn

        sims = np.dot(matrix, query_vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.similarity_threshold:
            entries[best_idx]["hits"] += 1
            self.stats["hits"] += 1
            return entries[best_idx]["response"], best_sim

        self.stats["misses"] += 1
        return None

    def put(self, tenant_id: str, query: str, query_vec: np.ndarray, response: Dict[str, Any]) -> None:
        entries = self.cache.setdefault(tenant_id, [])
        qn = np.linalg.norm(query_vec)
        if qn > 0:
            query_vec = query_vec / qn

        if len(entries) >= self.max_entries_per_tenant:
            # Evict LFU
            entries.sort(key=lambda x: (x["hits"], x["created_at"]))
            entries.pop(0)
            self.stats["evictions"] += 1

        entries.append({
            "query": query,
            "vector": query_vec,
            "response": response,
            "created_at": time.time(),
            "hits": 0
        })

    def forget_tenant(self, tenant_id: str) -> int:
        """GDPR Right to Erasure: removes all cached records for tenant."""
        if tenant_id in self.cache:
            count = len(self.cache[tenant_id])
            del self.cache[tenant_id]
            return count
        return 0

    def clear(self) -> None:
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
