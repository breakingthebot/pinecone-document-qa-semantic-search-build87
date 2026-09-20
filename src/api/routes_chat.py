"""
Multi-Turn Conversational Chat REST Routes.
Handles chat message exchange, query reformulation, session state persistence,
and cited assistant responses with guardrails.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from src.models.schema import (
    ChatRequest,
    ChatSessionResponse,
    QuestionAnsweringRequest,
)
from src.services.conversation_service import get_conversation_service
from src.services.qa_rag_service import QARagService

router = APIRouter(prefix="/api/chat", tags=["Conversational Chat"])
qa_service = QARagService()
conv_service = get_conversation_service()


@router.post("/message", response_model=ChatSessionResponse)
def send_chat_message(payload: ChatRequest) -> ChatSessionResponse:
    """
    Processes a multi-turn chat message:
    1. Resolves or creates conversation session.
    2. Reformulates follow-up queries using prior conversational context.
    3. Executes Pinecone hybrid vector search and synthesizes grounded answer.
    4. Evaluates RAG Triad faithfulness guardrails.
    5. Records message turns and returns full session state.
    """
    session_id = conv_service.get_or_create_session(
        session_id=payload.session_id,
        namespace=payload.namespace,
    )

    # 1. Reformulate question if pronoun-heavy or short follow-up
    reformulated_query = conv_service.reformulate_query(
        session_id=session_id,
        new_message=payload.message,
    )

    # 2. Record user message turn
    conv_service.add_user_message(
        session_id=session_id,
        message=payload.message,
    )

    # 3. Answer question via hybrid RAG pipeline
    qa_req = QuestionAnsweringRequest(
        question=reformulated_query,
        namespace=payload.namespace,
        alpha=payload.alpha,
        top_k=payload.top_k,
        category_filter=payload.category_filter,
    )
    qa_res = qa_service.answer_question(request=qa_req)

    # 4. Record assistant response turn
    conv_service.add_assistant_message(
        session_id=session_id,
        content=qa_res.answer,
        citations=qa_res.citations,
        guardrails=qa_res.guardrails,
    )

    session_state = conv_service.get_session(session_id=session_id)
    if not session_state:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation session state",
        )

    session_state.reformulated_query = reformulated_query
    return session_state


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_chat_session(session_id: str) -> ChatSessionResponse:
    """
    Retrieves complete multi-turn conversation history for the specified session.
    """
    session_state = conv_service.get_session(session_id=session_id)
    if not session_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found",
        )

    return session_state


@router.delete("/sessions/{session_id}")
def clear_chat_session(session_id: str) -> Dict[str, Any]:
    """
    Purges conversation history for the specified session.
    """
    deleted = conv_service.clear_session(session_id=session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found",
        )

    return {"ok": True, "cleared_session": session_id}
