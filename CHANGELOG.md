# Changelog — Build 87

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-15

### Added
- **Pinecone Vector Database Replica Engine (`MemoryPinecone`)**:
  - Implemented high-performance in-memory vector store matching official Pinecone index semantics.
  - Multi-namespace partition isolation (`default`, `knowledge-base`, `science-hub`, etc.).
  - Nearest-neighbor search supporting Cosine Similarity, Dot Product, and Euclidean distance.
  - Pinecone metadata filter engine supporting `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`, `$lte`, and `$and` operators.
  - Vector lifecycle operations: `upsert`, `query`, `fetch`, `delete`, and `describe_index_stats`.
- **Deterministic Dense Vector Embedder**:
  - 128-dimensional L2 unit-norm float vector generation from raw text.
  - Sublinear term frequency weighting and character n-gram hashing for zero-dependency semantic clustering.
- **Sentence-Window Sliding Chunker**:
  - Punctuation-aware sentence boundary preservation.
  - Configurable chunk size (400 characters) and sliding window overlap (80 characters).
  - Preserves metadata lineage (`doc_id`, `title`, `category`, `chunk_index`, custom tags).
- **Retrieval-Augmented Generation (RAG) QA Service**:
  - Natural language question embedding and top-k vector retrieval.
  - Grounded answer formulation with confidence scoring.
  - Hallucination prevention through automatic refusal below minimum score threshold.
  - Granular chunk-level citations (title, category, chunk index, score, snippet).
- **FastAPI REST API Suite**:
  - `GET /`: Discovery endpoint returning architecture, dimensions, and endpoint catalog.
  - `POST /api/documents`: Ingest document, chunk, embed, and index to Pinecone.
  - `GET /api/documents`: List ingested document catalog with namespace filtering.
  - `GET /api/documents/{doc_id}`: Retrieve single document metadata.
  - `DELETE /api/documents/{doc_id}`: Delete document and purge vector chunks from Pinecone.
  - `POST /api/documents/vectors/upsert`: Low-level raw vector upsert.
  - `POST /api/search/vectors`: Nearest-neighbor semantic search.
  - `POST /api/qa/ask`: End-to-end RAG question answering.
  - `GET /api/system/stats`: Index vector counts and namespace statistics.
  - `POST /api/system/reset`: Vector index purge for test isolation.
- **Automated Test Suite**:
  - 29 unit and integration tests across 5 test suites covering embedder, chunker, Pinecone engine, RAG pipeline, and REST API.
