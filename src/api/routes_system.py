"""
System and Vector Index Telemetry REST Routes.
Provides index statistics, namespace counts, and test state resets.
"""

from typing import Dict, Any
from fastapi import APIRouter

from src.models.schema import IndexStatsResponse
from src.engine.pinecone_client import get_pinecone_engine, reset_pinecone_engine

router = APIRouter(prefix="/api/system", tags=["System & Telemetry"])


@router.get("/stats", response_model=IndexStatsResponse)
def get_index_statistics() -> IndexStatsResponse:
    """
    Returns Pinecone index telemetry including vector dimensions, similarity metric,
    total vector count, and vector distribution across namespaces.
    """
    engine = get_pinecone_engine()
    return engine.describe_index_stats()


@router.post("/reset")
def reset_vector_index() -> Dict[str, Any]:
    """
    Purges all vector records across all namespaces for test isolation.
    """
    reset_pinecone_engine()
    return {
        "ok": True,
        "cleared": True,
        "message": "Pinecone vector index reset successfully",
    }
