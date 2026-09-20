"""
Guardrails and Answer Evaluation REST Routes.
Exposes standalone RAG Triad faithfulness, context relevance, and hallucination scoring.
"""

from typing import List
from pydantic import BaseModel, Field
from fastapi import APIRouter

from src.models.schema import Citation, GuardrailEvaluation
from src.services.guardrails_service import get_guardrails_service

router = APIRouter(prefix="/api/guardrails", tags=["Guardrails & Evaluation"])
guardrails_service = get_guardrails_service()


class GuardrailEvaluateRequest(BaseModel):
    """
    Payload for evaluating answer grounding against citations.
    """

    question: str = Field(..., min_length=1, description="Source query")
    answer: str = Field(..., min_length=1, description="Synthesized answer to verify")
    citations: List[Citation] = Field(default_factory=list, description="Candidate context citations")


@router.post("/evaluate", response_model=GuardrailEvaluation)
def evaluate_grounding(payload: GuardrailEvaluateRequest) -> GuardrailEvaluation:
    """
    Evaluates faithfulness, context relevance, and grounding for an answer given citations.
    """
    return guardrails_service.evaluate_answer(
        question=payload.question,
        answer=payload.answer,
        citations=payload.citations,
    )
