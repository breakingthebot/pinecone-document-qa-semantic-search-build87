"""
Unit tests for DocumentService and QARagService.
Tests document ingestion, semantic search, cited answer synthesis, and vector cleanup.
"""

from src.models.schema import (
    DocumentIngestRequest,
    VectorQueryRequest,
    QuestionAnsweringRequest,
)
from src.services.document_service import DocumentService
from src.services.qa_rag_service import QARagService
from src.engine.pinecone_client import get_pinecone_engine


def test_ingest_and_list_documents():
    """
    Test ingesting a document creates chunks, embeds them, and upserts them into Pinecone.
    """
    doc_svc = DocumentService()
    pinecone = get_pinecone_engine()

    content = (
        "Acme Corp Remote Work Policy. "
        "Employees are eligible to work remotely up to three days per week after completing ninety days of tenure. "
        "Core collaboration hours are 10:00 AM to 3:00 PM Eastern Time. "
        "Company provides an annual $500 home office equipment stipend."
    )

    doc_res = doc_svc.ingest_document(
        DocumentIngestRequest(
            title="Remote Work Guidelines",
            content=content,
            category="HR",
            author="People Operations",
            namespace="corporate",
            chunk_size=120,
            chunk_overlap=20,
        )
    )

    assert doc_res.id.startswith("doc_")
    assert doc_res.chunk_count >= 2
    assert doc_res.category == "HR"

    # Verify catalog contains document
    fetched = doc_svc.get_document(doc_res.id)
    assert fetched is not None
    assert fetched.title == "Remote Work Guidelines"

    # Verify Pinecone holds vectors in the corporate namespace
    stats = pinecone.describe_index_stats()
    assert stats.namespaces["corporate"]["vector_count"] == doc_res.chunk_count


def test_semantic_search_with_query_text():
    """
    Test vector nearest-neighbor query using natural language search phrase.
    """
    doc_svc = DocumentService()
    qa_svc = QARagService()

    doc_svc.ingest_document(
        DocumentIngestRequest(
            title="Travel Expense Guidelines",
            content="Employees traveling on company business are entitled to a $75 daily meal per diem. Flight bookings must be made in economy class.",
            category="Finance",
        )
    )

    search_res = qa_svc.semantic_search(
        VectorQueryRequest(
            query_text="What is the meal per diem rate for travel?",
            top_k=2,
        )
    )

    assert len(search_res.matches) >= 1
    top_match = search_res.matches[0]
    assert top_match.score > 0.12
    assert "daily meal per diem" in top_match.metadata["text"]


def test_qa_pipeline_accurate_answer_and_citations():
    """
    Test RAG question answering pipeline returns accurate answer with document citations.
    """
    doc_svc = DocumentService()
    qa_svc = QARagService()

    doc_svc.ingest_document(
        DocumentIngestRequest(
            title="Engineering Security Protocols",
            content=(
                "All production database access requires multi-factor hardware security keys. "
                "SSH access to cloud virtual machines is restricted to corporate VPN IP ranges. "
                "Production secrets must be rotated every ninety calendar days."
            ),
            category="Security",
            author="InfoSec Team",
        )
    )

    qa_res = qa_svc.answer_question(
        QuestionAnsweringRequest(
            question="How often must production secrets be rotated?",
            top_k=3,
        )
    )

    assert "Engineering Security Protocols" in qa_res.answer
    assert "ninety calendar days" in qa_res.answer
    assert qa_res.confidence_score > 0.30
    assert len(qa_res.citations) >= 1

    top_citation = qa_res.citations[0]
    assert top_citation.title == "Engineering Security Protocols"
    assert top_citation.category == "Security"
    assert "ninety calendar days" in top_citation.snippet


def test_qa_pipeline_category_filtering():
    """
    Test Q&A query restricted to a specific category only retrieves chunks from that category.
    """
    doc_svc = DocumentService()
    qa_svc = QARagService()

    doc_svc.ingest_document(
        DocumentIngestRequest(
            title="Benefits Handbook",
            content="Company dental coverage covers two cleanings per year at 100 percent.",
            category="HR",
        )
    )
    doc_svc.ingest_document(
        DocumentIngestRequest(
            title="Infrastructure Playbook",
            content="Backup clusters are verified every month using automated recovery drills.",
            category="DevOps",
        )
    )

    # Ask question filtered strictly to DevOps
    qa_res = qa_svc.answer_question(
        QuestionAnsweringRequest(
            question="Tell me about dental coverage",
            category_filter="DevOps",
        )
    )

    # Because dental coverage is HR, and we filtered to DevOps, it should not match dental
    assert "dental" not in qa_res.answer.lower()


def test_qa_pipeline_unrelated_question():
    """
    Test asking an unrelated question returns a graceful rejection with zero hallucinations.
    """
    doc_svc = DocumentService()
    qa_svc = QARagService()

    doc_svc.ingest_document(
        DocumentIngestRequest(
            title="Coffee Machine User Guide",
            content="Press the button to brew espresso. Descale the water tank every month.",
            category="Office",
        )
    )

    qa_res = qa_svc.answer_question(
        QuestionAnsweringRequest(
            question="What was the gross domestic product of France in 1848?",
            min_score_threshold=0.85,
        )
    )

    assert "could not find any relevant documentation" in qa_res.answer.lower()
    assert len(qa_res.citations) == 0
    assert qa_res.confidence_score == 0.0


def test_delete_document_purges_vectors():
    """
    Test deleting a document purges all chunk vectors from Pinecone.
    """
    doc_svc = DocumentService()
    pinecone = get_pinecone_engine()

    doc = doc_svc.ingest_document(
        DocumentIngestRequest(
            title="Temporary Onboarding Guide",
            content="Welcome to the team. Setup your laptop and create your email password.",
            category="HR",
            namespace="temp_ns",
        )
    )

    stats_before = pinecone.describe_index_stats()
    assert stats_before.namespaces["temp_ns"]["vector_count"] >= 1

    deleted = doc_svc.delete_document(doc.id)
    assert deleted is True
    assert doc_svc.get_document(doc.id) is None

    stats_after = pinecone.describe_index_stats()
    assert stats_after.namespaces["temp_ns"]["vector_count"] == 0
