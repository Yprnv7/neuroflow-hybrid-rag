import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_\-\.]{2,}\b")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """
    In-memory BM25Okapi lexical retrieval engine.
    Supports multi-tenant partition filtering.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len = 0.0
        self.term_freqs: List[Counter] = []
        self.doc_freqs: Counter = Counter()
        self.tenant_indices: Dict[str, List[int]] = {}

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = Counter()
        self.tenant_indices = {}

        total_tokens = 0
        for idx, chunk in enumerate(chunks):
            tokens = tokenize(chunk["text"])
            length = len(tokens)
            self.doc_lens.append(length)
            total_tokens += length

            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

            t_id = chunk.get("tenant_id", "default")
            self.tenant_indices.setdefault(t_id, []).append(idx)

        self.avg_doc_len = total_tokens / len(chunks) if chunks else 0.0

    def search(self, query: str, tenant_id: str = None, top_k: int = 50) -> List[Tuple[Dict[str, Any], float]]:
        if not self.chunks:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        candidate_indices = (
            self.tenant_indices.get(tenant_id, [])
            if tenant_id is not None
            else range(len(self.chunks))
        )

        n = len(self.chunks)
        scores = []

        for doc_idx in candidate_indices:
            score = 0.0
            dl = self.doc_lens[doc_idx]
            tf_dict = self.term_freqs[doc_idx]

            for term in q_tokens:
                df = self.doc_freqs.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_dict.get(term, 0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                score += idf * (numerator / denominator)

            if score > 0.0:
                scores.append((self.chunks[doc_idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
