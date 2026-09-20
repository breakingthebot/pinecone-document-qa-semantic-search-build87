"""
Pinecone Vector Search REST API Routes.
Exposes nearest-neighbor semantic search by vector or natural language query.
"""

from fastapi import APIRouter, HTTPException, status

from src.models.schema import (
    VectorQueryRequest,
    VectorQueryResponse,
    HybridQueryRequest,
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


@router.post("/hybrid", response_model=VectorQueryResponse)
def query_hybrid(payload: HybridQueryRequest) -> VectorQueryResponse:
    """
    Executes Pinecone hybrid search blending dense vector embeddings and BM25 sparse vectors
    using configurable alpha weighting (1.0 = dense semantic, 0.0 = sparse keyword).
    """
    try:
        return service.hybrid_search(request=payload)
    except ValueError as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ex),
        )
