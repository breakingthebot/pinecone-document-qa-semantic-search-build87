"""
Retrieval-Augmented Generation (RAG) and Semantic Search Service.
Executes vector nearest-neighbor search and synthesizes grounded answers with source citations.
"""

import time
import re
from typing import Dict, Any, List, Optional

from src.config import settings
from src.engine.pinecone_client import get_pinecone_engine
from src.engine.embedder import get_embedder
from src.engine.bm25_sparse import get_sparse_vectorizer
from src.services.guardrails_service import get_guardrails_service
from src.models.schema import (
    VectorQueryRequest,
    VectorQueryResponse,
    HybridQueryRequest,
    QuestionAnsweringRequest,
    QuestionAnsweringResponse,
    Citation,
)


class QARagService:
    """
    RAG pipeline service orchestrating semantic and hybrid search, answer synthesis, and guardrails.
    """

    def __init__(self) -> None:
        self.engine = get_pinecone_engine()
        self.embedder = get_embedder()
        self.sparse_vectorizer = get_sparse_vectorizer()
        self.guardrails_service = get_guardrails_service()

    def semantic_search(self, request: VectorQueryRequest) -> VectorQueryResponse:
        """
        Executes semantic vector search against Pinecone index.
        If query_text is supplied, generates dense embedding on-the-fly.
        """
        query_vec: List[float]

        if request.vector is not None:
            query_vec = request.vector
        elif request.query_text:
            query_vec = self.embedder.embed_text(request.query_text)
        else:
            raise ValueError("Either 'vector' or 'query_text' must be provided for search")

        ns = request.namespace or settings.DEFAULT_NAMESPACE

        return self.engine.query(
            vector=query_vec,
            top_k=request.top_k,
            namespace=ns,
            filter=request.filter,
            include_metadata=request.include_metadata,
            include_values=request.include_values,
        )

    def hybrid_search(self, request: HybridQueryRequest) -> VectorQueryResponse:
        """
        Executes Pinecone hybrid search blending dense semantic embeddings with BM25 sparse vectors.
        """
        dense_vec = self.embedder.embed_text(request.query_text)
        sparse_vec = self.sparse_vectorizer.encode_text(request.query_text)
        ns = request.namespace or settings.DEFAULT_NAMESPACE

        return self.engine.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            alpha=request.alpha,
            top_k=request.top_k,
            namespace=ns,
            filter=request.filter,
            include_metadata=request.include_metadata,
            include_values=False,
        )

    def answer_question(self, request: QuestionAnsweringRequest) -> QuestionAnsweringResponse:
        """
        Executes full RAG workflow:
        1. Encodes question into dense vector and BM25 sparse vector.
        2. Retrieves top-k candidate chunks from Pinecone using hybrid scoring.
        3. Filters by score threshold and metadata filters.
        4. Extracts relevant sentences and synthesizes answer with document citations.
        5. Computes RAG Triad faithfulness and hallucination guardrail metrics.
        """
        start_time = time.perf_counter()
        ns = request.namespace or settings.DEFAULT_NAMESPACE

        # Build Pinecone metadata filter
        query_filter: Dict[str, Any] = {}
        if request.metadata_filter:
            query_filter.update(request.metadata_filter)

        if request.category_filter:
            query_filter["category"] = request.category_filter

        active_filter = query_filter if query_filter else None

        # 1. Embed user question: dense + sparse vectors
        question_vector = self.embedder.embed_text(request.question)
        sparse_vector = self.sparse_vectorizer.encode_text(request.question)

        # 2. Retrieve nearest neighbor chunks from Pinecone using hybrid scoring
        search_res = self.engine.query(
            vector=question_vector,
            sparse_vector=sparse_vector,
            alpha=request.alpha,
            top_k=request.top_k or settings.DEFAULT_TOP_K,
            namespace=ns,
            filter=active_filter,
            include_metadata=True,
            include_values=False,
        )

        # 3. Filter by minimum score threshold
        qualifying_matches = [
            m for m in search_res.matches if m.score >= request.min_score_threshold
        ]

        # Handle case where no relevant chunks exist
        if not qualifying_matches:
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return QuestionAnsweringResponse(
                question=request.question,
                answer="I could not find any relevant documentation answering this question in the vector index.",
                confidence_score=0.0,
                citations=[],
                total_candidates_reviewed=len(search_res.matches),
                namespace=ns,
                processing_time_ms=duration_ms,
                guardrails=None,
            )

        # 4. Extract citations and synthesize answer
        citations: List[Citation] = []
        synthesized_points: List[str] = []

        # Extract keywords from question to highlight best sentences
        q_words = set(re.findall(r"\b[a-z0-9]+\b", request.question.lower()))

        for match in qualifying_matches:
            meta = match.metadata or {}
            chunk_text = meta.get("text", "")
            title = meta.get("title", "Unknown Document")
            cat = meta.get("category", "General")
            doc_id = meta.get("doc_id", "unknown")
            chunk_idx = meta.get("chunk_index", 0)

            # Find best sentence inside chunk that matches question tokens
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if s.strip()]
            best_sentence = sentences[0] if sentences else chunk_text
            best_overlap = -1

            for s in sentences:
                s_words = set(re.findall(r"\b[a-z0-9]+\b", s.lower()))
                overlap = len(q_words.intersection(s_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = s

            citations.append(
                Citation(
                    doc_id=doc_id,
                    title=title,
                    category=cat,
                    chunk_index=chunk_idx,
                    similarity_score=match.score,
                    snippet=best_sentence,
                )
            )
            synthesized_points.append(best_sentence)

        # Calculate confidence based on top similarity score
        top_score = qualifying_matches[0].score
        confidence = min(1.0, max(0.0, round(top_score, 2)))

        # Format synthesized answer
        answer_body = " ".join(dict.fromkeys(synthesized_points))
        formatted_answer = f"Based on {citations[0].title} ({citations[0].category}): {answer_body}"

        # 5. Evaluate Faithfulness and Hallucination Guardrails
        guardrail_eval = self.guardrails_service.evaluate_answer(
            question=request.question,
            answer=formatted_answer,
            citations=citations,
        )

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return QuestionAnsweringResponse(
            question=request.question,
            answer=formatted_answer,
            confidence_score=confidence,
            citations=citations,
            total_candidates_reviewed=len(search_res.matches),
            namespace=ns,
            processing_time_ms=duration_ms,
            guardrails=guardrail_eval,
        )
