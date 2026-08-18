"""Resolve pronouns and deictic references using recent conversation."""

from __future__ import annotations

import re
from typing import List

from memory.conversation_memory import ConversationEntry
from utils.logging_utils import setup_logging

logger = setup_logging("memory.log")


class PronounResolver:
    COURSE_RE = re.compile(r"\b([A-Z]{2,}\d{2,})\b")

    def resolve(self, query: str, recent: List[ConversationEntry]) -> str:
        """Attempt to rewrite `query` by replacing pronouns with the most recent entity.

        Strategy:
        - Find most recent course-like token (e.g., CS101) in recent user messages.
        - Otherwise find the most recent Capitalized token longer than 2 characters.
        - Replace occurrences of pronouns like 'its' with that entity.
        """
        if not recent:
            return query

        # gather candidates from recent entries (most recent first)
        candidates: List[str] = []
        for entry in reversed(recent):
            text = f"{entry.user_query} {entry.assistant_response}"
            for match in self.COURSE_RE.findall(text):
                candidates.append(match)
            # capitalized words
            for token in re.findall(r"\b([A-Z][a-z][A-Za-z0-9_\-]+)\b", text):
                candidates.append(token)
            if candidates:
                break

        if not candidates:
            return query

        entity = candidates[0]
        q = query
        # replace possessive pronouns
        q = re.sub(r"\bits\b", entity, q, flags=re.IGNORECASE)
        q = re.sub(r"\bit\b", entity, q, flags=re.IGNORECASE)
        q = re.sub(r"\btheir\b", entity, q, flags=re.IGNORECASE)
        q = re.sub(r"\bthey\b", entity, q, flags=re.IGNORECASE)
        q = re.sub(r"\bthis\b", entity, q, flags=re.IGNORECASE)
        q = re.sub(r"\bthat\b", entity, q, flags=re.IGNORECASE)
        logger.info("PronounResolver replaced pronoun(s) with %s for query: %s", entity, query)
        return q
