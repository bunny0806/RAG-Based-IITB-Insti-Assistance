import pytest

from memory.memory_manager import MemoryManager
from memory.conversation_memory import ConversationEntry


def test_followup_detection_and_pronoun_resolution():
    mgr = MemoryManager()
    sid = "session-1"
    mem = mgr.get_memory(sid)
    # add initial user message about CS101
    entry = ConversationEntry(user_query="Tell me about CS101", assistant_response="CS101 is Intro to CS.")
    mem.add_entry(entry)

    # follow-up question with pronoun
    raw = "What about its grading?"
    resolved = mgr.resolve_query(sid, raw)
    assert "CS101" in resolved or "cs101" in resolved.lower()


def test_summarization_and_context_building():
    mgr = MemoryManager()
    sid = "session-sum"
    mem = mgr.get_memory(sid)
    # add several entries to force summarization
    for i in range(8):
        mem.add_entry(ConversationEntry(user_query=f"Question {i}", assistant_response=f"Answer {i}"))

    context = mgr.build_context(sid)
    assert "recent_messages" in context
    assert isinstance(context["summary"], str)


def test_session_isolation_and_clear():
    mgr = MemoryManager()
    s1 = "s-A"
    s2 = "s-B"
    m1 = mgr.get_memory(s1)
    m2 = mgr.get_memory(s2)
    m1.add_entry(ConversationEntry(user_query="Hi", assistant_response="Hello"))
    assert m1.length() == 1
    assert m2.length() == 0

    mgr.clear_memory(s1)
    assert m1.length() == 0
