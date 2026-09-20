"""
Unit tests for Faithfulness and Hallucination Guardrails Service.
Verifies RAG Triad faithfulness scoring, context relevance, and ungrounded claim detection.
"""

from src.models.schema import Citation
from src.services.guardrails_service import GuardrailsService


def test_guardrails_grounded_answer():
    """
    Tests that an answer strictly supported by citation snippets receives high faithfulness.
    """
    guardrails = GuardrailsService(faithfulness_threshold=0.70)

    question = "How does quantum superposition work?"
    answer = "Based on Quantum Guide (Physics): Superposition allows quantum bits to exist in multiple states simultaneously."
    citations = [
        Citation(
            doc_id="doc_q1",
            title="Quantum Guide",
            category="Physics",
            chunk_index=0,
            similarity_score=0.92,
            snippet="Superposition allows quantum bits to exist in multiple states simultaneously.",
        )
    ]

    evaluation = guardrails.evaluate_answer(
        question=question,
        answer=answer,
        citations=citations,
    )

    assert evaluation.is_grounded is True
    assert evaluation.faithfulness_score >= 0.80
    assert evaluation.context_relevance_score > 0.0
    assert len(evaluation.verification_notes) > 0


def test_guardrails_hallucinated_answer():
    """
    Tests that an answer making unbacked claims receives low faithfulness and fails grounding.
    """
    guardrails = GuardrailsService(faithfulness_threshold=0.70)

    question = "What is FastAPI?"
    answer = "FastAPI was secretly developed by ancient civilizations in the year 3000 BC to fly UFOs across galaxies."
    citations = [
        Citation(
            doc_id="doc_fa1",
            title="FastAPI Overview",
            category="Web",
            chunk_index=0,
            similarity_score=0.45,
            snippet="FastAPI is a modern, high-performance web framework for building APIs with Python.",
        )
    ]

    evaluation = guardrails.evaluate_answer(
        question=question,
        answer=answer,
        citations=citations,
    )

    assert evaluation.is_grounded is False
    assert evaluation.faithfulness_score < 0.50
    assert any("potentially ungrounded" in note for note in evaluation.verification_notes)


def test_guardrails_empty_citations():
    """
    Tests that an answer without citations defaults to ungrounded.
    """
    guardrails = GuardrailsService()
    evaluation = guardrails.evaluate_answer(
        question="Can quantum computers factor RSA keys?",
        answer="I do not have sufficient context.",
        citations=[],
    )

    assert evaluation.is_grounded is False
    assert evaluation.faithfulness_score == 0.0
    assert evaluation.context_relevance_score == 0.0
