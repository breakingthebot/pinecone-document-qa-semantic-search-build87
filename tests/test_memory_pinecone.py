"""
Unit tests for MemoryPinecone vector index engine.
Verifies upsert, fetch, nearest-neighbor search, metrics, namespaces, and metadata filters.
"""

import pytest
import numpy as np
from src.engine.memory_pinecone import MemoryPinecone
from src.models.schema import VectorRecord


def test_upsert_and_fetch_vectors():
    """
    Test upserting vector records and fetching them back by ID.
    """
    engine = MemoryPinecone(dimension=4, metric="cosine")
    records = [
        VectorRecord(id="v1", values=[1.0, 0.0, 0.0, 0.0], metadata={"tag": "red"}),
        VectorRecord(id="v2", values=[0.0, 1.0, 0.0, 0.0], metadata={"tag": "blue"}),
    ]

    count = engine.upsert(records, namespace="colors")
    assert count == 2

    fetched = engine.fetch(["v1", "v2", "nonexistent"], namespace="colors")
    assert len(fetched) == 2
    assert fetched["v1"]["metadata"]["tag"] == "red"
    assert fetched["v2"]["metadata"]["tag"] == "blue"


def test_upsert_dimension_validation_failure():
    """
    Test upserting a vector with mismatched dimensions raises ValueError.
    """
    engine = MemoryPinecone(dimension=4)
    bad_record = [VectorRecord(id="v_bad", values=[1.0, 0.0], metadata={})]

    with pytest.raises(ValueError) as exc_info:
        engine.upsert(bad_record)

    assert "Vector dimension mismatch" in str(exc_info.value)


def test_cosine_query_ranking():
    """
    Test nearest neighbor query ranks vectors correctly by cosine similarity.
    """
    engine = MemoryPinecone(dimension=3, metric="cosine")
    records = [
        VectorRecord(id="target_match", values=[1.0, 0.0, 0.0], metadata={"name": "exact"}),
        VectorRecord(id="close_match", values=[0.8, 0.6, 0.0], metadata={"name": "close"}),
        VectorRecord(id="orthogonal", values=[0.0, 1.0, 0.0], metadata={"name": "ortho"}),
    ]
    engine.upsert(records)

    query_vec = [1.0, 0.0, 0.0]
    res = engine.query(vector=query_vec, top_k=2)

    assert len(res.matches) == 2
    assert res.matches[0].id == "target_match"
    assert abs(res.matches[0].score - 1.0) < 1e-3
    assert res.matches[1].id == "close_match"
    assert res.matches[1].score > 0.70


def test_dotproduct_and_euclidean_metrics():
    """
    Test Dot Product and Euclidean distance metrics.
    """
    engine_dot = MemoryPinecone(dimension=2, metric="dotproduct")
    engine_dot.upsert([
        VectorRecord(id="d1", values=[2.0, 3.0], metadata={}),
        VectorRecord(id="d2", values=[1.0, 1.0], metadata={}),
    ])
    dot_res = engine_dot.query(vector=[1.0, 2.0], top_k=1)
    # [1, 2] . [2, 3] = 2 + 6 = 8.0
    assert dot_res.matches[0].id == "d1"
    assert abs(dot_res.matches[0].score - 8.0) < 1e-3

    engine_euc = MemoryPinecone(dimension=2, metric="euclidean")
    engine_euc.upsert([
        VectorRecord(id="e1", values=[1.0, 1.0], metadata={}),
        VectorRecord(id="e2", values=[5.0, 5.0], metadata={}),
    ])
    euc_res = engine_euc.query(vector=[1.0, 1.0], top_k=1)
    # Distance = 0, score = 1 / (1 + 0) = 1.0
    assert euc_res.matches[0].id == "e1"
    assert abs(euc_res.matches[0].score - 1.0) < 1e-3


def test_namespace_isolation():
    """
    Test queries in namespace A never return records stored in namespace B.
    """
    engine = MemoryPinecone(dimension=2)
    engine.upsert([VectorRecord(id="doc_hr", values=[1.0, 0.0], metadata={})], namespace="hr_ns")
    engine.upsert([VectorRecord(id="doc_eng", values=[1.0, 0.0], metadata={})], namespace="eng_ns")

    hr_res = engine.query(vector=[1.0, 0.0], top_k=10, namespace="hr_ns")
    assert len(hr_res.matches) == 1
    assert hr_res.matches[0].id == "doc_hr"

    eng_res = engine.query(vector=[1.0, 0.0], top_k=10, namespace="eng_ns")
    assert len(eng_res.matches) == 1
    assert eng_res.matches[0].id == "doc_eng"


def test_metadata_filtering():
    """
    Test Pinecone metadata filter operators ($eq, $in, $gt).
    """
    engine = MemoryPinecone(dimension=2)
    records = [
        VectorRecord(id="r1", values=[1.0, 0.0], metadata={"category": "HR", "level": 2}),
        VectorRecord(id="r2", values=[0.9, 0.1], metadata={"category": "Engineering", "level": 4}),
        VectorRecord(id="r3", values=[0.8, 0.2], metadata={"category": "HR", "level": 5}),
    ]
    engine.upsert(records)

    # Filter by category $eq
    res_hr = engine.query(vector=[1.0, 0.0], top_k=10, filter={"category": "HR"})
    assert len(res_hr.matches) == 2
    assert set([m.id for m in res_hr.matches]) == {"r1", "r3"}

    # Filter by level $gt
    res_gt = engine.query(vector=[1.0, 0.0], top_k=10, filter={"level": {"$gt": 3}})
    assert len(res_gt.matches) == 2
    assert set([m.id for m in res_gt.matches]) == {"r2", "r3"}

    # Filter with $in
    res_in = engine.query(vector=[1.0, 0.0], top_k=10, filter={"category": {"$in": ["Engineering"]}})
    assert len(res_in.matches) == 1
    assert res_in.matches[0].id == "r2"


def test_delete_operations():
    """
    Test deleting vectors by ID, by filter, and deleting all records in a namespace.
    """
    engine = MemoryPinecone(dimension=2)
    engine.upsert([
        VectorRecord(id="v1", values=[1.0, 0.0], metadata={"env": "dev"}),
        VectorRecord(id="v2", values=[0.0, 1.0], metadata={"env": "dev"}),
        VectorRecord(id="v3", values=[0.5, 0.5], metadata={"env": "prod"}),
    ])

    # Delete single vector by ID
    del_count = engine.delete(ids=["v1"])
    assert del_count == 1
    assert len(engine.fetch(["v1"])) == 0

    # Delete by filter
    del_filter_count = engine.delete(filter={"env": "prod"})
    assert del_filter_count == 1
    assert len(engine.fetch(["v3"])) == 0

    # Delete all remaining
    del_all = engine.delete(delete_all=True)
    assert del_all == 1
    stats = engine.describe_index_stats()
    assert stats.total_vector_count == 0


def test_describe_index_stats():
    """
    Test index telemetry reports vector counts per namespace.
    """
    engine = MemoryPinecone(index_name="qa-index", dimension=128, metric="cosine")
    engine.upsert([VectorRecord(id="v1", values=[0.1] * 128, metadata={})], namespace="ns1")
    engine.upsert([VectorRecord(id="v2", values=[0.2] * 128, metadata={})], namespace="ns2")
    engine.upsert([VectorRecord(id="v3", values=[0.3] * 128, metadata={})], namespace="ns2")

    stats = engine.describe_index_stats()
    assert stats.index_name == "qa-index"
    assert stats.dimension == 128
    assert stats.metric == "cosine"
    assert stats.total_vector_count == 3
    assert stats.namespaces["ns1"]["vector_count"] == 1
    assert stats.namespaces["ns2"]["vector_count"] == 2
