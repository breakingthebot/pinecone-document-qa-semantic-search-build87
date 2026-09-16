"""
FastAPI Application Entrypoint for Build 87: Pinecone Document Q&A.
Demonstrates dense vector embeddings, Pinecone index namespace partitioning,
nearest-neighbor semantic search, metadata filtering, and cited RAG answer synthesis.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.engine.pinecone_client import get_pinecone_engine
from src.api.routes_documents import router as documents_router
from src.api.routes_search import router as search_router
from src.api.routes_qa import router as qa_router
from src.api.routes_system import router as system_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager for startup and shutdown hooks.
    """
    engine = get_pinecone_engine()
    yield


app = FastAPI(
    title="Build 87: Pinecone Document Q&A",
    description="Vector database semantic search and Retrieval-Augmented Generation (RAG) powered by Pinecone dense indexing, metadata filtering, and cited answer synthesis.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST sub-routers
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(qa_router)
app.include_router(system_router)


@app.get("/", tags=["Discovery"])
def root_discovery() -> Dict[str, Any]:
    """
    Root discovery endpoint providing architecture details, vector dimensions, and RAG guide.
    """
    engine = get_pinecone_engine()
    stats = engine.describe_index_stats()

    return {
        "build": 87,
        "service": "Build 87: Pinecone Document Q&A",
        "technology": "Pinecone Vector Database & FastAPI",
        "category": "Databases - Vector/Search",
        "index_configuration": {
            "index_name": settings.PINECONE_INDEX_NAME,
            "dimension": settings.PINECONE_DIMENSION,
            "metric": settings.PINECONE_METRIC,
            "default_namespace": settings.DEFAULT_NAMESPACE,
        },
        "rag_pipeline": {
            "chunking": f"Sentence-window with {settings.DEFAULT_CHUNK_SIZE} char size and {settings.DEFAULT_CHUNK_OVERLAP} char overlap",
            "embedding": f"Normalized {settings.PINECONE_DIMENSION}-dimensional dense float vectors",
            "retrieval": f"Cosine nearest-neighbor search with top_k={settings.DEFAULT_TOP_K} and metadata filtering",
            "synthesis": "Extractive and abstractive answer formulation with granular document chunk citations",
        },
        "endpoints": {
            "document_ingest": "/api/documents",
            "semantic_search": "/api/search/vectors",
            "rag_question_answering": "/api/qa/ask",
            "index_telemetry": "/api/system/stats",
        },
        "telemetry": stats.model_dump(),
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
