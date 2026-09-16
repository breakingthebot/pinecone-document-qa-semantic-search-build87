"""
Integration tests for Build 87 FastAPI endpoints.
Validates document ingestion, vector querying, QA generation, and system operations.
"""

from fastapi.testclient import TestClient


def test_root_discovery_endpoint(client: TestClient):
    """
    Tests that the root endpoint returns 200 OK and valid service discovery metadata.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["build"] == 87
    assert "Pinecone" in data["technology"]
    assert "endpoints" in data


def test_document_ingest_and_retrieval_flow(client: TestClient):
    """
    Tests the complete document lifecycle: ingest, fetch catalog, get single doc, and delete.
    """
    # 1. Ingest document
    payload = {
        "title": "Quantum Computing Fundamentals",
        "content": (
            "Quantum computers utilize quantum bits or qubits to perform calculations. "
            "Superposition allows qubits to exist in multiple states simultaneously. "
            "Quantum entanglement enables qubits to be correlated with one another instantaneously. "
            "These properties provide exponential speedups for specialized optimization problems."
        ),
        "category": "physics",
        "author": "Dr. Sarah Lin",
        "namespace": "science-hub",
        "metadata": {
            "department": "research",
            "year": 2026,
        },
    }

    ingest_res = client.post("/api/documents", json=payload)
    assert ingest_res.status_code == 201
    ingest_data = ingest_res.json()
    assert ingest_data["chunk_count"] > 0
    doc_id = ingest_data["id"]
    assert doc_id is not None

    # 2. List documents
    list_res = client.get("/api/documents")
    assert list_res.status_code == 200
    catalog = list_res.json()
    assert len(catalog) == 1
    assert catalog[0]["id"] == doc_id
    assert catalog[0]["title"] == "Quantum Computing Fundamentals"

    # 3. Retrieve specific document
    get_res = client.get(f"/api/documents/{doc_id}")
    assert get_res.status_code == 200
    fetched_doc = get_res.json()
    assert fetched_doc["id"] == doc_id
    assert fetched_doc["title"] == "Quantum Computing Fundamentals"
    assert fetched_doc["chunk_count"] == ingest_data["chunk_count"]

    # 4. Check system stats reflect the upserted vectors
    stats_res = client.get("/api/system/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_vector_count"] >= ingest_data["chunk_count"]
    assert "science-hub" in stats_data["namespaces"]

    # 5. Delete document
    delete_res = client.delete(f"/api/documents/{doc_id}")
    assert delete_res.status_code == 200
    assert delete_res.json()["ok"] is True
    assert delete_res.json()["deleted"] == doc_id

    # 6. Verify deleted from catalog
    get_after_delete = client.get(f"/api/documents/{doc_id}")
    assert get_after_delete.status_code == 404


def test_upsert_raw_vector_endpoint(client: TestClient):
    """
    Tests upserting raw vector embeddings directly into a target namespace.
    """
    vector_values = [0.1] * 128
    payload = {
        "namespace": "custom-namespace",
        "vectors": [
            {
                "id": "raw-vec-001",
                "values": vector_values,
                "metadata": {
                    "source": "custom_embedding_model",
                    "score_tier": "gold",
                },
            }
        ],
    }

    upsert_res = client.post("/api/documents/vectors/upsert", json=payload)
    assert upsert_res.status_code == 200
    upsert_data = upsert_res.json()
    assert upsert_data["upserted_count"] == 1
    assert upsert_data["namespace"] == "custom-namespace"

    # Search the custom namespace for the upserted vector
    search_payload = {
        "vector": vector_values,
        "namespace": "custom-namespace",
        "top_k": 3,
        "include_metadata": True,
    }
    search_res = client.post("/api/search/vectors", json=search_payload)
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert len(search_data["matches"]) >= 1
    assert search_data["matches"][0]["id"] == "raw-vec-001"
    assert search_data["matches"][0]["metadata"]["score_tier"] == "gold"


def test_semantic_search_with_metadata_filters(client: TestClient):
    """
    Tests semantic vector query using text and Pinecone metadata filtering operators.
    """
    # Ingest document for Engineering
    doc_eng = {
        "title": "Kubernetes Architecture",
        "content": "Kubernetes orchestrates containerized workloads across compute clusters with resilient pods.",
        "category": "devops",
        "namespace": "knowledge-base",
        "metadata": {
            "department": "engineering",
        },
    }
    client.post("/api/documents", json=doc_eng)

    # Ingest document for Marketing
    doc_mkt = {
        "title": "Brand Positioning Strategy",
        "content": "Brand positioning elevates market awareness through consistent messaging and customer advocacy.",
        "category": "branding",
        "namespace": "knowledge-base",
        "metadata": {
            "department": "marketing",
        },
    }
    client.post("/api/documents", json=doc_mkt)

    # Search with department filter for engineering
    filter_eng_query = {
        "query_text": "container orchestration pods",
        "namespace": "knowledge-base",
        "top_k": 5,
        "filter": {
            "department": "engineering",
        },
    }
    search_eng_res = client.post("/api/search/vectors", json=filter_eng_query)
    assert search_eng_res.status_code == 200
    matches_eng = search_eng_res.json()["matches"]
    assert len(matches_eng) > 0
    for match in matches_eng:
        assert match["metadata"]["department"] == "engineering"

    # Search with department filter for marketing
    filter_mkt_query = {
        "query_text": "market positioning and awareness",
        "namespace": "knowledge-base",
        "top_k": 5,
        "filter": {
            "department": "marketing",
        },
    }
    search_mkt_res = client.post("/api/search/vectors", json=filter_mkt_query)
    assert search_mkt_res.status_code == 200
    matches_mkt = search_mkt_res.json()["matches"]
    assert len(matches_mkt) > 0
    for match in matches_mkt:
        assert match["metadata"]["department"] == "marketing"


def test_qa_ask_endpoint_with_grounded_answer_and_citations(client: TestClient):
    """
    Tests RAG QA endpoint, verifying synthesized answer, confidence score, and citations.
    """
    doc_payload = {
        "title": "FastAPI Framework Guide",
        "content": (
            "FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+. "
            "It is built on top of Starlette for routing and Pydantic for data validation. "
            "Automatic interactive documentation is generated using Swagger UI and ReDoc."
        ),
        "category": "frameworks",
        "namespace": "developer-docs",
        "metadata": {
            "department": "backend",
        },
    }
    client.post("/api/documents", json=doc_payload)

    qa_payload = {
        "question": "What is FastAPI built on top of?",
        "namespace": "developer-docs",
        "top_k": 3,
        "min_score_threshold": 0.10,
    }

    qa_res = client.post("/api/qa/ask", json=qa_payload)
    assert qa_res.status_code == 200
    qa_data = qa_res.json()

    assert qa_data["question"] == "What is FastAPI built on top of?"
    assert "Starlette" in qa_data["answer"] or "Pydantic" in qa_data["answer"]
    assert qa_data["confidence_score"] > 0.0
    assert len(qa_data["citations"]) > 0

    citation = qa_data["citations"][0]
    assert citation["title"] == "FastAPI Framework Guide"
    assert citation["category"] == "frameworks"
    assert citation["similarity_score"] > 0.0
    assert len(citation["snippet"]) > 0


def test_qa_ask_endpoint_refusal_when_no_relevant_chunk(client: TestClient):
    """
    Tests that the QA endpoint returns a polite refusal when similarity threshold is not met.
    """
    qa_payload = {
        "question": "What is the capital city of ancient Atlantis in year 4000 BC?",
        "namespace": "empty-namespace",
        "top_k": 3,
        "min_score_threshold": 0.50,
    }

    qa_res = client.post("/api/qa/ask", json=qa_payload)
    assert qa_res.status_code == 200
    qa_data = qa_res.json()

    assert "could not find any relevant documentation" in qa_data["answer"].lower()
    assert len(qa_data["citations"]) == 0
    assert qa_data["confidence_score"] == 0.0


def test_system_stats_and_reset_endpoints(client: TestClient):
    """
    Tests resetting the in-memory Pinecone index via the system management API.
    """
    # Ingest a document
    doc_payload = {
        "title": "Temporary Doc",
        "content": "This document will be cleared during index reset with enough characters.",
        "category": "temp",
        "namespace": "staging",
        "metadata": {
            "department": "ops",
        },
    }
    client.post("/api/documents", json=doc_payload)

    # Check stats before reset
    stats_before = client.get("/api/system/stats").json()
    assert stats_before["total_vector_count"] > 0

    # Reset system
    reset_res = client.post("/api/system/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["cleared"] is True

    # Check stats after reset
    stats_after = client.get("/api/system/stats").json()
    assert stats_after["total_vector_count"] == 0
    assert len(stats_after["namespaces"]) == 0
