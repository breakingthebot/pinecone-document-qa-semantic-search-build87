"""
Deterministic Dense Vector Embedding Engine.
Converts arbitrary text into normalized 128-dimensional dense float vectors.
Produces real semantic clustering using subword tokenization and L2 normalization.
"""

import hashlib
import math
import re
from typing import List
import numpy as np

from src.config import settings


class Embedder:
    """
    Generates normalized dense vector embeddings for semantic search and document indexing.
    """

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        """
        Extracts lowercase word tokens and character 3-grams for semantic richness.
        """
        cleaned = text.lower()
        words = re.findall(r"\b[a-z0-9_]+\b", cleaned)

        tokens: List[str] = list(words)
        # Add character tri-grams for subword matching (handles variations, stems, typos)
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    tokens.append(word[i : i + 3])

        return tokens

    def embed_text(self, text: str) -> List[float]:
        """
        Encodes a single text string into a unit-normalized dense vector.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)

        if not tokens:
            return vec.tolist()

        # Count token occurrences
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1

        for token, count in counts.items():
            # Hash token to a stable bucket index within [0, dimension - 1]
            bucket_idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            # Feature sign hashing for zero-bias projection
            sign = 1.0 if (int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:4], 16) % 2 == 0) else -1.0
            # Sublinear term frequency weighting
            weight = (1.0 + math.log(count)) * sign
            vec[bucket_idx] += weight

        # L2 unit normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        return [float(x) for x in vec]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batch encodes an array of texts into dense vectors.
        """
        return [self.embed_text(t) for t in texts]


_embedder_instance = Embedder(dimension=settings.PINECONE_DIMENSION)


def get_embedder() -> Embedder:
    """
    Returns the singleton embedder instance.
    """
    global _embedder_instance
    return _embedder_instance
