"""
Document Question Answering (RAG) REST Routes.
Synthesizes answers grounded strictly in retrieved vector context with source citations.
"""

from fastapi import APIRouter, HTTPException, status

from src.models.schema import (
    QuestionAnsweringRequest,
    QuestionAnsweringResponse,
)
from src.services.qa_rag_service import QARagService

router = APIRouter(prefix="/api/qa", tags=["Question Answering (RAG)"])
service = QARagService()


@router.post("/ask", response_model=QuestionAnsweringResponse)
def ask_question(payload: QuestionAnsweringRequest) -> QuestionAnsweringResponse:
    """
    RAG Question-Answering Endpoint:
    1. Embeds the user question into a dense query vector.
    2. Retrieves top-k nearest candidate chunks from Pinecone.
    3. Synthesizes a factual answer citing document source, chunk index, and snippet.
    """
    try:
        return service.answer_question(request=payload)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Q&A synthesis failed: {str(ex)}",
        )
