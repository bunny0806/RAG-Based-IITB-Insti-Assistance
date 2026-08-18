"""Conversation memory primitives and in-memory conversation store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from utils.logging_utils import setup_logging
from observability import trace_stage

logger = setup_logging("memory.log")


@dataclass
class ConversationEntry:
    user_query: str
    assistant_response: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_document_ids: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence: Optional[float] = None
    # Optional metadata for memory-aware workflows
    original_query: Optional[str] = None
    resolved_query: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    summary_used: Optional[str] = None
    followup_detected: bool = False
    pronoun_resolved: bool = False
    memory_summary_length: int = 0
    recent_context_size: int = 0


class ConversationMemory:
    """Thread-safe conversation memory log.

    Stores a chronological list of ConversationEntry objects and provides
    utilities for retrieving recent history and compacting older history.
    """

    def __init__(self) -> None:
        self._entries: List[ConversationEntry] = []
        self._lock = Lock()

    def add_entry(self, entry: ConversationEntry) -> None:
        with trace_stage("storage"):
            with self._lock:
                self._entries.append(entry)
                logger.info("Added conversation entry: %s", entry.user_query)

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            logger.info("Cleared conversation memory.")

    def list_entries(self) -> List[ConversationEntry]:
        with self._lock:
            return list(self._entries)

    def recent(self, n: int = 5) -> List[ConversationEntry]:
        with self._lock:
            return list(self._entries[-n:])

    def length(self) -> int:
        with self._lock:
            return len(self._entries)
