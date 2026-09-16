"""
Unit tests for ChunkingService.
Verifies sentence boundary detection, sliding window chunk creation, and metadata enrichment.
"""

from src.services.chunking_service import ChunkingService


def test_split_sentences():
    """
    Test splitting text into complete sentences at punctuation boundaries.
    """
    chunker = ChunkingService()
    text = "Pinecone is a managed vector store. It handles billions of vectors! Can it scale efficiently? Yes."

    sentences = chunker.split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Pinecone is a managed vector store."
    assert sentences[1] == "It handles billions of vectors!"
    assert sentences[2] == "Can it scale efficiently?"
    assert sentences[3] == "Yes."


def test_chunk_document_boundaries():
    """
    Test chunking long document partitions text into multiple chunks while keeping sentences intact.
    """
    chunker = ChunkingService()
    content = (
        "Chapter 1: The foundation of vector databases. "
        "High-dimensional embeddings represent semantic concepts mathematically. "
        "Cosine similarity measures the angular distance between dense vectors. "
        "Chapter 2: Scaling vector search with approximate nearest neighbors. "
        "Hierarchical Navigable Small World graphs provide sub-millisecond retrieval. "
        "Inverted file indexes cluster vectors into Voronoi cells. "
        "Chapter 3: Metadata filtering and hybrid search algorithms."
    )

    chunks = chunker.chunk_document(
        doc_id="doc_test_1",
        title="Vector Database Architecture",
        content=content,
        category="Engineering",
        source_url="https://docs.pinecone.io",
        author="Cloud Architect",
        chunk_size=150,
        chunk_overlap=30,
    )

    assert len(chunks) >= 2
    for c in chunks:
        assert c.doc_id == "doc_test_1"
        assert c.metadata["title"] == "Vector Database Architecture"
        assert c.metadata["category"] == "Engineering"
        assert len(c.text) > 0


def test_chunk_document_short_text():
    """
    Test short text that fits within chunk_size generates exactly one chunk without redundancy.
    """
    chunker = ChunkingService()
    content = "This is a brief memo on annual benefits."

    chunks = chunker.chunk_document(
        doc_id="doc_memo",
        title="Benefits Memo",
        content=content,
        category="HR",
        chunk_size=400,
        chunk_overlap=50,
    )

    assert len(chunks) == 1
    assert chunks[0].id == "doc_memo#chunk_0"
    assert chunks[0].text == content
    assert chunks[0].chunk_index == 0
