"""
Pinecone Client Factory and Engine Singleton.
Provides get_pinecone_engine() with transparent fallback to MemoryPinecone.
"""

from typing import Optional
from src.config import settings
from src.engine.memory_pinecone import MemoryPinecone

_engine_instance: Optional[MemoryPinecone] = None


def get_pinecone_engine() -> MemoryPinecone:
    """
    Returns the singleton Pinecone index engine instance.
    Defaults to MemoryPinecone for high-speed local testing with zero cloud credentials.
    """
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = MemoryPinecone(
            index_name=settings.PINECONE_INDEX_NAME,
            dimension=settings.PINECONE_DIMENSION,
            metric=settings.PINECONE_METRIC,
        )

    return _engine_instance


def reset_pinecone_engine() -> None:
    """
    Resets all vector index namespaces and telemetry for test isolation.
    """
    global _engine_instance

    if _engine_instance is not None:
        _engine_instance.reset_all()
    else:
        _engine_instance = MemoryPinecone(
            index_name=settings.PINECONE_INDEX_NAME,
            dimension=settings.PINECONE_DIMENSION,
            metric=settings.PINECONE_METRIC,
        )
