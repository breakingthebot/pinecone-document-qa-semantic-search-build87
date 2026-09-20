"""
BM25 Sparse Vectorizer Engine for Pinecone Hybrid Search.
Converts arbitrary text into high-dimensional sparse representations
with integer coordinate indices and TF-IDF / BM25 term weights.
"""

import math
import re
import hashlib
from typing import List, Dict, Tuple
from src.models.schema import SparseValues


class BM25SparseVectorizer:
    """
    Computes sparse vector representations for Pinecone hybrid search.
    Maps token stems to deterministic 32-bit integer indices.
    """

    def __init__(self, max_indices: int = 1000000, k1: float = 1.5, b: float = 0.75) -> None:
        self.max_indices = max_indices
        self.k1 = k1
        self.b = b
        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
            "is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "this", "that", "these", "those", "it", "its"
        }

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into cleaned lowercase alphanumeric terms excluding common stop words.
        """
        raw_words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        tokens = [w for w in raw_words if len(w) >= 2 and w not in self.stopwords]
        return tokens

    def _token_to_index(self, token: str) -> int:
        """
        Hashes token to a stable positive integer index within [1, max_indices].
        """
        md5_digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(md5_digest[:8], 16) % self.max_indices + 1
        return index

    def encode_text(self, text: str, avg_doc_len: float = 50.0) -> SparseValues:
        """
        Encodes text into a Pinecone SparseValues object containing parallel indices and values.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return SparseValues(indices=[], values=[])

        doc_len = len(tokens)
        freq_map: Dict[str, int] = {}
        for t in tokens:
            freq_map[t] = freq_map.get(t, 0) + 1

        # Accumulate scores per hashed index
        sparse_map: Dict[int, float] = {}
        for token, count in freq_map.items():
            idx = self._token_to_index(token)
            # BM25 term frequency formula
            num = count * (self.k1 + 1.0)
            denom = count + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, avg_doc_len)))
            tf_bm25 = num / denom
            # Rare term / length weighting heuristic
            rarity_weight = 1.0 + math.log(max(1.0, len(token)))
            weight = tf_bm25 * rarity_weight
            sparse_map[idx] = sparse_map.get(idx, 0.0) + weight

        # Sort by indices for deterministic Pinecone sparse vector ordering
        sorted_pairs: List[Tuple[int, float]] = sorted(sparse_map.items(), key=lambda pair: pair[0])

        # L2 normalize sparse vector values
        raw_values = [p[1] for p in sorted_pairs]
        norm = math.sqrt(sum(v * v for v in raw_values))
        if norm > 1e-6:
            normalized_values = [round(v / norm, 5) for v in raw_values]
        else:
            normalized_values = [round(v, 5) for v in raw_values]

        return SparseValues(
            indices=[p[0] for p in sorted_pairs],
            values=normalized_values,
        )

    def compute_sparse_similarity(self, query_sparse: SparseValues, doc_sparse: SparseValues) -> float:
        """
        Computes the inner product between two sparse vectors in linear time.
        """
        if not query_sparse.indices or not doc_sparse.indices:
            return 0.0

        # Build dictionary for document sparse vector
        doc_dict = dict(zip(doc_sparse.indices, doc_sparse.values))

        score = 0.0
        for q_idx, q_val in zip(query_sparse.indices, query_sparse.values):
            if q_idx in doc_dict:
                score += q_val * doc_dict[q_idx]

        return float(min(1.0, max(0.0, score)))


_sparse_vectorizer_instance = BM25SparseVectorizer()


def get_sparse_vectorizer() -> BM25SparseVectorizer:
    """
    Returns the singleton BM25 sparse vectorizer instance.
    """
    global _sparse_vectorizer_instance
    return _sparse_vectorizer_instance
