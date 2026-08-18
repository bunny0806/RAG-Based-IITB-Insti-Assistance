"""High-level MemoryManager exposing APIs for the application and UI."""

from __future__ import annotations

from threading import Lock
from typing import Optional

from memory.session_store import SessionStore
from memory.context_builder import ContextBuilder
from memory.followup_detector import FollowupDetector
from memory.pronoun_resolver import PronounResolver
from memory.conversation_memory import ConversationEntry
from utils.logging_utils import setup_logging
from observability import trace_stage

logger = setup_logging("memory.log")

_MANAGER_CACHE: dict[str, "MemoryManager"] = {}
_MANAGER_LOCK = Lock()


class MemoryManager:
    """Wrapper managing per-session ConversationMemory and memory utilities."""

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        self.store = SessionStore(storage_dir)
        self.context_builder = ContextBuilder()
        self.detector = FollowupDetector()
        self.resolver = PronounResolver()

    def get_memory(self, session_id: str):
        return self.store.get(session_id)

    def clear_memory(self, session_id: str) -> None:
        mem = self.store.get(session_id)
        mem.clear()

    def build_context(self, session_id: str) -> dict:
        mem = self.store.get(session_id)
        return self.context_builder.build(mem)

    def resolve_query(self, session_id: str, query: str) -> str:
        with trace_stage("memory_resolution"):
            mem = self.store.get(session_id)
            recent = mem.recent(8)
            followup_detected = self.detector.is_followup(query, recent)
            logger.info("Follow-up detection for session %s: %s", session_id, followup_detected)
            if followup_detected:
                resolved = self.resolver.resolve(query, recent)
                pronoun_resolved = resolved.strip().lower() != query.strip().lower()
                logger.info("Pronoun resolution for session %s: %s", session_id, pronoun_resolved)
                logger.info("Resolved query for session %s: %s -> %s", session_id, query, resolved)
                return resolved
            logger.info("Resolved query for session %s: %s -> %s", session_id, query, query)
            return query


def get_memory_manager(storage_dir: Optional[str] = None) -> MemoryManager:
    """Return a shared MemoryManager instance for a given storage directory."""
    cache_key = storage_dir or "default"
    with _MANAGER_LOCK:
        if cache_key not in _MANAGER_CACHE:
            _MANAGER_CACHE[cache_key] = MemoryManager(storage_dir=storage_dir)
        return _MANAGER_CACHE[cache_key]
