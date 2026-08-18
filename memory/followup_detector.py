"""Detect whether a query is a follow-up and needs context."""

from __future__ import annotations

import re
from typing import List

from memory.conversation_memory import ConversationEntry


class FollowupDetector:
    PRONOUNS = {"it", "its", "they", "their", "them", "this", "that", "those", "these", "one", "ones"}

    def is_followup(self, query: str, recent: List[ConversationEntry]) -> bool:
        q = query.lower().strip()
        # Explicit continuation phrasing can be a follow-up without a pronoun.
        if recent and ("what about" in q or "tell me more" in q or "explain more" in q):
            return True
        # short queries that refer with pronouns or deictic phrases are follow-ups
        if any(p in q.split() for p in self.PRONOUNS):
            return True
        if len(q.split()) <= 3 and not re.search(r"\b(what|who|when|where|why|how)\b", q):
            # e.g., 'What about grading?' (short, but containing 'about')
            if "about" in q or "what about" in q:
                return True
            return False
        # otherwise check if query lacks a clear named entity (capitalized or code)
        if not re.search(r"[A-Z]{2,}\d{1,}|[A-Z][a-z]{2,}", query):
            # could be follow-up if recent context exists
            return len(recent) > 0
        return False
