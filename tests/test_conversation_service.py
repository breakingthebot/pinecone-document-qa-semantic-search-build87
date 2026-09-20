"""
Unit tests for ConversationService.
Verifies multi-turn session persistence, query reformulation, and history pruning.
"""

from src.models.schema import Citation
from src.services.conversation_service import ConversationService


def test_session_lifecycle_and_messages():
    """
    Tests creating a chat session and recording message turns.
    """
    service = ConversationService(max_history_turns=5)
    sess_id = service.get_or_create_session(namespace="test-chat")

    service.add_user_message(sess_id, "What is Pinecone?")
    service.add_assistant_message(
        session_id=sess_id,
        content="Pinecone is a cloud-native vector database.",
        citations=[],
    )

    state = service.get_session(sess_id)
    assert state is not None
    assert state.session_id == sess_id
    assert state.turn_count == 2
    assert state.messages[0].role == "user"
    assert state.messages[1].role == "assistant"


def test_query_reformulation_for_pronoun_followup():
    """
    Tests that a follow-up with pronouns incorporates the prior document topic.
    """
    service = ConversationService()
    sess_id = service.get_or_create_session()

    # Initial turn with a citation referencing "Pinecone Architecture"
    service.add_user_message(sess_id, "Tell me about vector indexing.")
    service.add_assistant_message(
        session_id=sess_id,
        content="Pinecone supports serverless vector indexes.",
        citations=[
            Citation(
                doc_id="doc_pc1",
                title="Pinecone Architecture",
                category="Databases",
                chunk_index=0,
                similarity_score=0.88,
                snippet="Pinecone indexes partition vectors across isolated namespaces.",
            )
        ],
    )

    # Follow-up question containing pronoun 'it'
    followup = "How does it handle multi-tenancy?"
    reformulated = service.reformulate_query(sess_id, followup)

    assert "Pinecone Architecture" in reformulated
    assert "How does it handle multi-tenancy?" in reformulated

    # Non-pronoun standalone query should remain untouched
    standalone = "Write a Python function to sort numbers"
    assert service.reformulate_query(sess_id, standalone) == standalone


def test_session_context_pruning():
    """
    Tests that history is pruned to avoid exceeding the maximum turn budget.
    """
    service = ConversationService(max_history_turns=2)
    sess_id = service.get_or_create_session()

    # Add 4 turns (8 messages)
    for i in range(4):
        service.add_user_message(sess_id, f"Question #{i}")
        service.add_assistant_message(sess_id, f"Answer #{i}")

    state = service.get_session(sess_id)
    # max_history_turns = 2, so max 4 messages kept
    assert state.turn_count == 4
    assert state.messages[-1].content == "Answer #3"


def test_clear_session():
    """
    Tests purging a conversation session.
    """
    service = ConversationService()
    sess_id = service.get_or_create_session()
    service.add_user_message(sess_id, "Hello")

    assert service.get_session(sess_id) is not None
    cleared = service.clear_session(sess_id)
    assert cleared is True
    assert service.get_session(sess_id) is None
