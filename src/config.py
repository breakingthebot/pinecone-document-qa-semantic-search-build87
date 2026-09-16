"""
Build 87 Configuration Settings.
Provides environment-backed configurations for Pinecone vector database and RAG pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory resolution
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()


class Settings:
    """
    Application and Pinecone runtime configuration.
    """

    # Server Settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    APP_ENV: str = os.getenv("APP_ENV", "development")

    # Pinecone Vector Database Settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "mock-pinecone-api-key")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "document-qa-index")
    PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "128"))
    PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "cosine").lower()

    # Engine Selection: embedded pure-Python replica vs live Pinecone API
    USE_EMBEDDED_ENGINE: bool = os.getenv("USE_EMBEDDED_ENGINE", "true").lower() in ("true", "1", "yes")

    # RAG Chunking & Retrieval Defaults
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "400"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "80"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "4"))
    DEFAULT_NAMESPACE: str = os.getenv("DEFAULT_NAMESPACE", "default")


settings = Settings()
