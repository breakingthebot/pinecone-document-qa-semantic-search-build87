"""
Multi-Turn Conversational Memory and Query Reformulation Service.
Manages chat session state, context budgeting, coreference resolution,
and query reformulation for follow-up conversational turns.
"""

import uuid
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional
from src.models.schema import (
    ChatMessage,
    ChatRequest,
    ChatSessionResponse,
    Citation,
    GuardrailEvaluation,
)


class ConversationService:
    """
    Manages multi-turn conversation sessions and resolves conversational context for vector search.
    """

    def __init__(self, max_history_turns: int = 10) -> None:
        self.max_history_turns = max_history_turns
        # Session state: session_id -> List[ChatMessage]
        self.sessions: Dict[str, List[ChatMessage]] = {}
        # Session metadata: session_id -> {"namespace": str}
        self.session_meta: Dict[str, Dict[str, str]] = {}

    def _now_iso(self) -> str:
        """
        Returns UTC timestamp in ISO 8601 string format.
        """
        return datetime.now(timezone.utc).isoformat()

    def get_or_create_session(self, session_id: Optional[str] = None, namespace: str = "default") -> str:
        """
        Initializes a new session if not present, or returns existing session ID.
        """
        if not session_id or session_id not in self.sessions:
            new_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
            self.sessions[new_id] = []
            self.session_meta[new_id] = {"namespace": namespace}
            return new_id

        return session_id

    def reformulate_query(self, session_id: str, new_message: str) -> str:
        """
        Reformulates conversational follow-up questions containing pronouns (it, they, this, that)
        into standalone semantic queries utilizing previous turn context.
        """
        history = self.sessions.get(session_id, [])
        if not history:
            return new_message

        # Check if new message contains pronouns or is a short follow-up
        pronoun_pattern = r"\b(it|its|they|them|their|this|that|these|those|the former|the latter)\b"
        is_pronoun_query = bool(re.search(pronoun_pattern, new_message, re.IGNORECASE))
        is_short_followup = len(new_message.split()) <= 4

        if not (is_pronoun_query or is_short_followup):
            return new_message

        # Find the last topic from previous assistant or user message
        last_turn = history[-1]
        topic = ""

        # Check if last turn had citations
        if last_turn.citations:
            topic = last_turn.citations[0].title
        else:
            # Extract prominent nouns/words from previous content
            words = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", last_turn.content)
            if words:
                topic = " ".join(words[:2])

        if topic:
            return f"Regarding {topic}: {new_message}"

        return new_message

    def add_user_message(self, session_id: str, message: str) -> None:
        """
        Appends user message turn to session history.
        """
        msg = ChatMessage(
            role="user",
            content=message,
            timestamp=self._now_iso(),
        )
        self.sessions[session_id].append(msg)
        self._prune_history(session_id)

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        citations: Optional[List[Citation]] = None,
        guardrails: Optional[GuardrailEvaluation] = None,
    ) -> None:
        """
        Appends assistant response turn with attached citations and guardrail metrics.
        """
        msg = ChatMessage(
            role="assistant",
            content=content,
            citations=citations or [],
            guardrails=guardrails,
            timestamp=self._now_iso(),
        )
        self.sessions[session_id].append(msg)
        self._prune_history(session_id)

    def _prune_history(self, session_id: str) -> None:
        """
        Enforces maximum conversation context budget by trimming older turns.
        """
        max_messages = self.max_history_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def get_session(self, session_id: str) -> Optional[ChatSessionResponse]:
        """
        Retrieves complete conversation session state.
        """
        if session_id not in self.sessions:
            return None

        messages = self.sessions[session_id]
        meta = self.session_meta.get(session_id, {"namespace": "default"})

        return ChatSessionResponse(
            session_id=session_id,
            messages=messages,
            namespace=meta.get("namespace", "default"),
            turn_count=len(messages),
        )

    def clear_session(self, session_id: str) -> bool:
        """
        Clears conversation history for a given session.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            if session_id in self.session_meta:
                del self.session_meta[session_id]
            return True
        return False


_conversation_instance = ConversationService()


def get_conversation_service() -> ConversationService:
    """
    Returns the singleton conversation service instance.
    """
    global _conversation_instance
    return _conversation_instance
