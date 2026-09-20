# Changelog — Build 87

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-20

### Added
- **Pinecone Hybrid Search Engine**:
  - Implemented BM25 sparse vectorizer with integer coordinate hashing and normalized term weights.
  - Added hybrid retrieval query execution in `MemoryPinecone` and `QARagService` with linear $\alpha$ weighting (`alpha * dense + (1 - alpha) * sparse`).
  - Added `POST /api/search/hybrid` endpoint for executing blended vector searches.
- **Multi-Turn Conversational Memory & Query Reformulation**:
  - Implemented `ConversationService` for managing persistent multi-turn chat sessions (`session_id`).
  - Implemented coreference query reformulation that incorporates previous turn document topics into pronoun follow-up questions.
  - Added conversational API endpoints: `POST /api/chat/message`, `GET /api/chat/sessions/{id}`, and `DELETE /api/chat/sessions/{id}`.
- **RAG Triad Faithfulness & Hallucination Guardrails**:
  - Implemented `GuardrailsService` evaluating answer claim grounding against retrieved citation snippets.
  - Computes Faithfulness score ($0.0 - 1.0$), Context Relevance score ($0.0 - 1.0$), Grounding boolean status, and verification notes.
  - Added `POST /api/guardrails/evaluate` endpoint.
- **Interactive Web Showcase UI Dashboard**:
  - Mounted modern glassmorphic dashboard on `/dashboard` and static assets on `/static`.
  - Added 4 interactive operational tabs: Document Studio, Hybrid Search Playground, Conversational RAG Chat, and Index Telemetry Inspector.
- **Extended Test Suite**:
  - Expanded test coverage from 29 to 43 passing automated tests across 8 test suites.

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
