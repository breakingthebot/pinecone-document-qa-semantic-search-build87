"""
In-Memory Pinecone Vector Database Index Replica.
Implements multi-namespace vector indexing, dense similarity scoring (Cosine, Dot Product, Euclidean),
k-Nearest Neighbor (k-NN) top_k retrieval, and Pinecone metadata filtering operators.
"""

import threading
from typing import Dict, Any, List, Optional
import numpy as np

from src.config import settings
from src.models.schema import (
    VectorRecord,
    ScoredMatch,
    VectorQueryResponse,
    IndexStatsResponse,
)


class MemoryPinecone:
    """
    In-memory replica of a Pinecone vector index supporting multi-namespace storage,
    nearest-neighbor similarity search, and Pinecone-compatible metadata filtering.
    """

    def __init__(
        self,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
    ) -> None:
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.dimension = dimension or settings.PINECONE_DIMENSION
        self.metric = (metric or settings.PINECONE_METRIC).lower()
        # Map: namespace_name -> { vector_id: {"id": str, "values": np.ndarray, "metadata": dict} }
        self.namespaces: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.lock = threading.RLock()

    def _get_namespace_store(self, namespace: str) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves or initializes a namespace vector dictionary.
        """
        ns = namespace if namespace else "default"
        if ns not in self.namespaces:
            self.namespaces[ns] = {}
        return self.namespaces[ns]

    def upsert(self, vectors: List[VectorRecord], namespace: str = "default") -> int:
        """
        Upserts an array of vector records into the specified namespace.
        Supports dense values, optional sparse values, and metadata.
        """
        with self.lock:
            store = self._get_namespace_store(namespace)
            upserted = 0

            for record in vectors:
                vec_arr = np.array(record.values, dtype=np.float32)

                # Validate vector dimension matches index configuration
                if len(vec_arr) != self.dimension:
                    raise ValueError(
                        f"Vector dimension mismatch: expected {self.dimension}, got {len(vec_arr)}"
                    )

                store[record.id] = {
                    "id": record.id,
                    "values": vec_arr,
                    "sparse_values": record.sparse_values,
                    "metadata": dict(record.metadata),
                }
                upserted += 1

            return upserted

    def fetch(self, ids: List[str], namespace: str = "default") -> Dict[str, Dict[str, Any]]:
        """
        Retrieves vector records by ID from the specified namespace.
        """
        with self.lock:
            store = self._get_namespace_store(namespace)
            results: Dict[str, Dict[str, Any]] = {}

            for vid in ids:
                if vid in store:
                    item = store[vid]
                    results[vid] = {
                        "id": item["id"],
                        "values": item["values"].tolist(),
                        "sparse_values": item.get("sparse_values"),
                        "metadata": dict(item["metadata"]),
                    }

            return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: str = "default",
        filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Deletes vector records from a namespace by ID list, metadata filter, or entire namespace.
        """
        with self.lock:
            store = self._get_namespace_store(namespace)
            deleted_count = 0

            if delete_all:
                deleted_count = len(store)
                store.clear()
                return deleted_count

            if ids:
                for vid in ids:
                    if vid in store:
                        del store[vid]
                        deleted_count += 1

            if filter:
                matched_ids = [
                    vid
                    for vid, item in store.items()
                    if self._matches_filter(item["metadata"], filter)
                ]
                for vid in matched_ids:
                    del store[vid]
                    deleted_count += 1

            return deleted_count

    def query(
        self,
        vector: List[float],
        sparse_vector: Optional[Any] = None,
        alpha: float = 1.0,
        top_k: int = 4,
        namespace: str = "default",
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> VectorQueryResponse:
        """
        Executes k-nearest neighbors (k-NN) vector search against vectors in the specified namespace.
        Supports pure dense (alpha=1.0), pure sparse (alpha=0.0), or hybrid weighted scoring.
        Applies metadata filters before ranking.
        """
        from src.engine.bm25_sparse import get_sparse_vectorizer

        query_vec = np.array(vector, dtype=np.float32)
        sparse_engine = get_sparse_vectorizer()

        with self.lock:
            store = self._get_namespace_store(namespace)
            candidates: List[Tuple[float, Dict[str, Any]]] = []

            for vid, item in store.items():
                # Apply metadata filter if specified
                if filter and not self._matches_filter(item["metadata"], filter):
                    continue

                # Compute dense similarity
                dense_score = self._compute_similarity(query_vec, item["values"])

                # Compute sparse similarity if sparse vector is provided and item has sparse values
                if sparse_vector is not None and item.get("sparse_values") is not None:
                    sparse_score = sparse_engine.compute_sparse_similarity(
                        query_sparse=sparse_vector,
                        doc_sparse=item["sparse_values"],
                    )
                else:
                    sparse_score = 0.0

                # Compute hybrid weighted score
                if sparse_vector is not None and alpha < 1.0:
                    combined_score = (alpha * dense_score) + ((1.0 - alpha) * sparse_score)
                else:
                    combined_score = dense_score

                candidates.append((combined_score, item))

            # Rank candidates descending by similarity score
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = candidates[:top_k]

            matches: List[ScoredMatch] = []
            for score, item in top_candidates:
                match_record = ScoredMatch(
                    id=item["id"],
                    score=round(float(score), 4),
                    metadata=dict(item["metadata"]) if include_metadata else None,
                    values=item["values"].tolist() if include_values else None,
                    sparse_values=item.get("sparse_values"),
                )
                matches.append(match_record)

            return VectorQueryResponse(
                matches=matches,
                namespace=namespace,
            )

    def _compute_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Computes vector similarity score according to configured metric.
        """
        if self.metric == "cosine":
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            if norm_u < 1e-6 or norm_v < 1e-6:
                return 0.0
            return float(np.dot(u, v) / (norm_u * norm_v))

        elif self.metric == "dotproduct":
            return float(np.dot(u, v))

        elif self.metric == "euclidean":
            dist = float(np.linalg.norm(u - v))
            return 1.0 / (1.0 + dist)

        # Default fallback to cosine
        return float(np.dot(u, v))

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Evaluates Pinecone metadata filter operators ($eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, $and, $or).
        """
        for key, condition in filter_dict.items():
            if key == "$and":
                if not all(self._matches_filter(metadata, sub) for sub in condition):
                    return False
                continue

            if key == "$or":
                if not any(self._matches_filter(metadata, sub) for sub in condition):
                    return False
                continue

            field_val = metadata.get(key)

            if isinstance(condition, dict):
                for op, expected in condition.items():
                    if op == "$eq" and field_val != expected:
                        return False
                    elif op == "$ne" and field_val == expected:
                        return False
                    elif op == "$gt" and not (field_val is not None and field_val > expected):
                        return False
                    elif op == "$gte" and not (field_val is not None and field_val >= expected):
                        return False
                    elif op == "$lt" and not (field_val is not None and field_val < expected):
                        return False
                    elif op == "$lte" and not (field_val is not None and field_val <= expected):
                        return False
                    elif op == "$in" and field_val not in expected:
                        return False
                    elif op == "$nin" and field_val in expected:
                        return False
            else:
                if field_val != condition:
                    return False

        return True

    def describe_index_stats(self) -> IndexStatsResponse:
        """
        Returns index telemetry, dimension, metric, and vector counts per namespace.
        """
        with self.lock:
            ns_stats: Dict[str, Dict[str, int]] = {}
            total = 0

            for ns, store in self.namespaces.items():
                cnt = len(store)
                ns_stats[ns] = {"vector_count": cnt}
                total += cnt

            return IndexStatsResponse(
                index_name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                total_vector_count=total,
                namespaces=ns_stats,
                engine="MemoryPinecone-1.0.0",
            )

    def reset_all(self) -> None:
        """
        Clears all namespaces and vectors from the index.
        """
        with self.lock:
            self.namespaces.clear()
