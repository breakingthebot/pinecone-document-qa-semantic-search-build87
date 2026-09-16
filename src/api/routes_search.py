"""
Pinecone Vector Search REST API Routes.
Exposes nearest-neighbor semantic search by vector or natural language query.
"""

from fastapi import APIRouter, HTTPException, status

from src.models.schema import (
    VectorQueryRequest,
    VectorQueryResponse,
)
from src.services.qa_rag_service import QARagService

router = APIRouter(prefix="/api/search", tags=["Vector Search"])
service = QARagService()


@router.post("/vectors", response_model=VectorQueryResponse)
def query_vectors(payload: VectorQueryRequest) -> VectorQueryResponse:
    """
    Executes Pinecone vector nearest-neighbor search.
    Accepts either an explicit float vector array or a query_text string to embed on-the-fly.
    """
    try:
        return service.semantic_search(request=payload)
    except ValueError as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ex),
        )
