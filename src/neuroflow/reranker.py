import math
from typing import Any, Dict, List, Tuple


class CrossEncoderReranker:
    """
    Reranks candidate chunks with cross-encoder/ms-marco-MiniLM-L-6-v2
    or lexical-positional token alignment fallback.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", fallback: bool = True):
        self.model_name = model_name
        self.fallback = fallback
        self._model = None

        if not fallback:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(model_name)
            except Exception:
                self.fallback = True

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not candidates:
            return []

        if not self.fallback and self._model is not None:
            pairs = [[query, c["text"]] for c in candidates]
            raw_scores = self._model.predict(pairs, batch_size=64, show_progress_bar=False)
            scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw_scores]
        else:
            q_words = set(query.lower().split())
            scores = []
            for c in candidates:
                d_words = c["text"].lower().split()
                if not d_words:
                    scores.append(0.0)
                    continue
                intersection = sum(1 for w in d_words if w in q_words)
                jaccard = intersection / max(1, len(q_words | set(d_words)))
                coverage = intersection / max(1, len(q_words))
                scores.append(round(0.6 * coverage + 0.4 * jaccard, 4))

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
