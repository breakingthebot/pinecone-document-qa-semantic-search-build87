"""
FastAPI Application Entrypoint for Build 87: Pinecone Document Q&A.
Demonstrates dense vector embeddings, Pinecone index namespace partitioning,
nearest-neighbor semantic search, metadata filtering, and cited RAG answer synthesis.
"""

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.engine.pinecone_client import get_pinecone_engine
from src.api.routes_documents import router as documents_router
from src.api.routes_search import router as search_router
from src.api.routes_qa import router as qa_router
from src.api.routes_system import router as system_router
from src.api.routes_chat import router as chat_router
from src.api.routes_guardrails import router as guardrails_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager for startup and shutdown hooks.
    """
    engine = get_pinecone_engine()
    yield


app = FastAPI(
    title="Build 87: Pinecone Document Q&A & Hybrid RAG",
    description="Enterprise vector database semantic and hybrid search powered by Pinecone dense indexing, BM25 sparse vectors, conversational memory, and faithfulness guardrails.",
    version="1.1.0",
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
app.include_router(chat_router)
app.include_router(guardrails_router)
app.include_router(system_router)

# Mount Static Dashboard Assets
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/dashboard", tags=["Showcase UI"])
def get_dashboard() -> FileResponse:
    """
    Serves the interactive Pinecone Document Q&A & Hybrid RAG Web Dashboard.
    """
    index_html = static_dir / "index.html"
    return FileResponse(index_html)


@app.get("/", tags=["Discovery"])
def root_discovery() -> Dict[str, Any]:
    """
    Root discovery endpoint providing architecture details, vector dimensions, and RAG guide.
    """
    engine = get_pinecone_engine()
    stats = engine.describe_index_stats()

    return {
        "build": 87,
        "service": "Build 87: Pinecone Document Q&A & Hybrid RAG",
        "technology": "Pinecone Vector Database & FastAPI",
        "category": "Databases - Vector/Search",
        "version": "1.1.0",
        "index_configuration": {
            "index_name": settings.PINECONE_INDEX_NAME,
            "dimension": settings.PINECONE_DIMENSION,
            "metric": settings.PINECONE_METRIC,
            "default_namespace": settings.DEFAULT_NAMESPACE,
        },
        "rag_pipeline": {
            "chunking": f"Sentence-window with {settings.DEFAULT_CHUNK_SIZE} char size and {settings.DEFAULT_CHUNK_OVERLAP} char overlap",
            "dense_embedding": f"Normalized {settings.PINECONE_DIMENSION}-dimensional dense float vectors",
            "sparse_embedding": "BM25 token-weighted sparse vectors with integer coordinate hashing",
            "hybrid_retrieval": "Linear blending with configurable alpha (0.0 BM25 to 1.0 Dense Cosine)",
            "conversational_memory": "Multi-turn session tracking with automatic coreference resolution & query reformulation",
            "guardrails": "RAG Triad faithfulness and hallucination verification metrics",
        },
        "endpoints": {
            "web_dashboard": "/dashboard",
            "document_ingest": "/api/documents",
            "semantic_search": "/api/search/vectors",
            "hybrid_search": "/api/search/hybrid",
            "rag_question_answering": "/api/qa/ask",
            "conversational_chat": "/api/chat/message",
            "guardrails_evaluation": "/api/guardrails/evaluate",
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
