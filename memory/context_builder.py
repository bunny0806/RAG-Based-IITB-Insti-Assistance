"""Build memory-aware context payload for retrieval and generation."""

from __future__ import annotations

from typing import List

from memory.conversation_memory import ConversationMemory, ConversationEntry
from memory.summarizer import Summarizer


class ContextBuilder:
    """Compose an active context from recent entries and a compressed summary."""

    def __init__(self, summary_chars: int = 400, recent_turns: int = 5) -> None:
        self.summarizer = Summarizer()
        self.summary_chars = summary_chars
        self.recent_turns = recent_turns

    def build(self, memory: ConversationMemory) -> dict:
        recent = memory.recent(self.recent_turns)
        all_entries = memory.list_entries()
        # older entries are those except recent
        older = all_entries[: max(0, len(all_entries) - len(recent))]
        summary = self.summarizer.summarize(older, max_chars=self.summary_chars)
        # compose active context
        active_messages = [f"USER: {e.user_query}\nASSISTANT: {e.assistant_response}" for e in recent]
        summary_length = len(summary)
        recent_context_size = sum(len(m) for m in active_messages)
        conversation_length = len(all_entries)
        total_memory_size = summary_length + recent_context_size
        context = {
            "summary": summary,
            "recent_messages": active_messages,
            "active_context_size": total_memory_size,
            "summary_length": summary_length,
            "recent_context_size": recent_context_size,
            "conversation_length": conversation_length,
            "compression_ratio": (summary_length / total_memory_size) if total_memory_size else 0.0,
        }
        return context
