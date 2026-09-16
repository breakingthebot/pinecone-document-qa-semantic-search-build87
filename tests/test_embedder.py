"""
Unit tests for Embedder.
Verifies vector dimensionality, L2 unit-norm scaling, repeatability, and semantic similarity.
"""

import numpy as np
from src.engine.embedder import Embedder


def test_embedder_dimension():
    """
    Test embedding generation produces vectors matching the configured dimension (128).
    """
    embedder = Embedder(dimension=128)
    vec = embedder.embed_text("Pinecone is a cloud-native vector database.")

    assert len(vec) == 128
    assert isinstance(vec[0], float)


def test_embedder_l2_normalization():
    """
    Test embedding vectors have an L2 norm of approximately 1.0 (unit vectors).
    """
    embedder = Embedder(dimension=128)
    vec = embedder.embed_text("Semantic search uses dense vector embeddings.")

    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-4


def test_embedder_deterministic():
    """
    Test identical text produces identical vector values.
    """
    embedder = Embedder(dimension=128)
    text = "Machine learning models transform natural language into embeddings."

    vec1 = embedder.embed_text(text)
    vec2 = embedder.embed_text(text)

    assert vec1 == vec2


def test_embedder_semantic_similarity():
    """
    Test semantically related texts have higher cosine similarity than unrelated texts.
    """
    embedder = Embedder(dimension=128)

    text_target = "Company policy allows employees to work remotely from home."
    text_related = "Remote work and home office arrangements are permitted by corporate guidelines."
    text_unrelated = "Quantum mechanics governs the behavior of subatomic particles."

    v_target = np.array(embedder.embed_text(text_target))
    v_related = np.array(embedder.embed_text(text_related))
    v_unrelated = np.array(embedder.embed_text(text_unrelated))

    sim_related = float(np.dot(v_target, v_related))
    sim_unrelated = float(np.dot(v_target, v_unrelated))

    assert sim_related > sim_unrelated
    assert sim_related > 0.15


def test_embedder_empty_text():
    """
    Test embedding empty text returns a zero vector without crashing.
    """
    embedder = Embedder(dimension=128)
    vec = embedder.embed_text("")

    assert len(vec) == 128
    assert all(x == 0.0 for x in vec)
