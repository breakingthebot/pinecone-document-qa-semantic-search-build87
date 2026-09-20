"""
Unit tests for Pinecone Hybrid Search and BM25 Sparse Vectorizer.
Verifies sparse vector generation, coordinate hashing, and alpha-weighted hybrid retrieval.
"""

from src.engine.bm25_sparse import BM25SparseVectorizer
from src.services.document_service import DocumentService
from src.services.qa_rag_service import QARagService
from src.models.schema import (
    DocumentIngestRequest,
    HybridQueryRequest,
)


def test_bm25_sparse_vectorizer_encoding():
    """
    Tests that the BM25 vectorizer produces positive integer indices and normalized float values.
    """
    vectorizer = BM25SparseVectorizer()
    sparse_vec = vectorizer.encode_text("Pinecone provides serverless vector indexes.")

    assert len(sparse_vec.indices) > 0
    assert len(sparse_vec.indices) == len(sparse_vec.values)
    # All indices must be positive integers
    assert all(isinstance(idx, int) and idx > 0 for idx in sparse_vec.indices)
    # Values should be non-negative floats
    assert all(val > 0.0 for val in sparse_vec.values)


def test_bm25_sparse_similarity_computation():
    """
    Tests that texts sharing distinctive keywords produce higher sparse similarity.
    """
    vectorizer = BM25SparseVectorizer()
    query_sparse = vectorizer.encode_text("Kubernetes cluster pods container")
    match_sparse = vectorizer.encode_text("Deploying pods inside a Kubernetes container cluster")
    unrelated_sparse = vectorizer.encode_text("Baking chocolate chip cookies with organic butter")

    sim_match = vectorizer.compute_sparse_similarity(query_sparse, match_sparse)
    sim_unrelated = vectorizer.compute_sparse_similarity(query_sparse, unrelated_sparse)

    assert sim_match > 0.30
    assert sim_unrelated == 0.0


def test_hybrid_search_alpha_weighting():
    """
    Tests hybrid search retrieval with dense, sparse, and balanced alpha values.
    """
    doc_service = DocumentService()
    qa_service = QARagService()

    # Ingest technical docs with specific acronyms
    doc_service.ingest_document(
        DocumentIngestRequest(
            title="PostgreSQL WAL Protocols",
            content="Write-Ahead Logging ensures relational atomicity. WAL segments are archived every hour.",
            category="Databases",
            namespace="hybrid-test",
        )
    )
    doc_service.ingest_document(
        DocumentIngestRequest(
            title="Network Security Firewalls",
            content="Stateful packet inspection inspects ingress and egress network gateway traffic.",
            category="Security",
            namespace="hybrid-test",
        )
    )

    # 1. Pure sparse keyword query (alpha = 0.0) for exact term 'WAL'
    sparse_res = qa_service.hybrid_search(
        HybridQueryRequest(
            query_text="WAL segments",
            namespace="hybrid-test",
            alpha=0.0,
            top_k=2,
        )
    )
    assert len(sparse_res.matches) >= 1
    assert "WAL" in sparse_res.matches[0].metadata["text"]

    # 2. Pure dense query (alpha = 1.0)
    dense_res = qa_service.hybrid_search(
        HybridQueryRequest(
            query_text="relational database crash recovery logging",
            namespace="hybrid-test",
            alpha=1.0,
            top_k=2,
        )
    )
    assert len(dense_res.matches) >= 1
    assert "PostgreSQL" in dense_res.matches[0].metadata["title"]

    # 3. Balanced hybrid query (alpha = 0.70)
    hybrid_res = qa_service.hybrid_search(
        HybridQueryRequest(
            query_text="Write-Ahead Logging WAL",
            namespace="hybrid-test",
            alpha=0.70,
            top_k=2,
        )
    )
    assert len(hybrid_res.matches) >= 1
    assert hybrid_res.matches[0].score > 0.15
