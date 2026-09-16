"""
Pytest configuration and shared fixtures for Build 87.
Automatically resets the Pinecone in-memory index before every test.
"""

import pytest
from fastapi.testclient import TestClient

from src.engine.pinecone_client import reset_pinecone_engine
from src.main import app


@pytest.fixture(autouse=True)
def reset_vector_index_state():
    """
    Purges all vector records and namespaces before each test case for clean test isolation.
    """
    reset_pinecone_engine()
    yield
    reset_pinecone_engine()


@pytest.fixture
def client():
    """
    FastAPI TestClient fixture for exercising HTTP endpoints.
    """
    with TestClient(app) as test_client:
        yield test_client
