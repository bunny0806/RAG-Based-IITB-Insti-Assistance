"""Simple conversation summarizer for compressing older messages."""

from __future__ import annotations

from typing import List

from memory.conversation_memory import ConversationEntry, ConversationMemory
from utils.logging_utils import setup_logging

logger = setup_logging("memory.log")


class Summarizer:
    """Summarize older conversation entries into a shorter text.

    This is a lightweight, deterministic summarizer suitable for on-device
    compression. It extracts repeated nouns and course-like tokens as facts.
    """

    def summarize(self, entries: List[ConversationEntry], max_chars: int = 500) -> str:
        if not entries:
            return ""
        # collect candidate tokens: uppercase course codes and capitalized words
        facts = []
        seen = set()
        for e in entries:
            text = f"{e.user_query}. {e.assistant_response}"
            # naive split tokens
            for token in text.split():
                t = token.strip(".,?!)(:;\"'")
                if not t:
                    continue
                if t.isupper() and any(ch.isdigit() for ch in t):
                    if t not in seen:
                        facts.append(t)
                        seen.add(t)
                elif t[0].isupper() and len(t) > 2:
                    if t not in seen:
                        facts.append(t)
                        seen.add(t)
            if len(" ".join(facts)) > max_chars:
                break

        summary = "; ".join(facts)
        if not summary:
            # fallback: join first sentences until max_chars
            texts = [f"{e.user_query} {e.assistant_response}" for e in entries]
            joined = " \n ".join(texts)
            return joined[:max_chars]
        logger.info("Summarizer produced %d-char summary.", len(summary))
        return summary[:max_chars]
