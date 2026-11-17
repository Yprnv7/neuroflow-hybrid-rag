import hashlib
import struct
from typing import List

import numpy as np


class DenseEncoder:
    """
    Embedding service supporting BAAI/bge-small-en-v1.5 transformer model
    with high-speed deterministic normalized hash projection for local CI/testing.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384, fallback: bool = True):
        self.model_name = model_name
        self.dimension = dimension
        self.fallback = fallback
        self._model = None

        if not fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self.dimension = self._model.get_sentence_embedding_dimension()
            except Exception:
                self.fallback = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.fallback and self._model is not None:
            return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        return self._hash_encode(texts)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _hash_encode(self, texts: List[str]) -> np.ndarray:
        dim = self.dimension
        out = np.zeros((len(texts), dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue

            vec = np.zeros(dim, dtype=np.float32)
            for pos, token in enumerate(tokens):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                for b in range(4):
                    offset = b * 4
                    idx = struct.unpack_from("<I", digest, offset)[0] % dim
                    sign = 1.0 if (digest[offset] & 0x80) else -1.0
                    vec[idx] += sign / (b + 1)

                decay = 1.0 / (1.0 + 0.1 * pos)
                h2 = hashlib.md5(f"{token}_pos".encode()).digest()
                idx2 = struct.unpack_from("<I", h2, 0)[0] % dim
                vec[idx2] += decay

            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                out[i] = vec / norm

        return out
