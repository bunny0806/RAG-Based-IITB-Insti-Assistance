"""Session-scoped memory store to isolate conversation memories per Streamlit session."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from memory.conversation_memory import ConversationMemory
from utils.logging_utils import setup_logging

logger = setup_logging("memory.log")


class SessionStore:
    """Manage ConversationMemory instances per session id.

    Optionally persists sessions to disk under a given directory.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._store: Dict[str, ConversationMemory] = {}
        self._lock = Lock()
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get(self, session_id: str) -> ConversationMemory:
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ConversationMemory()
            return self._store[session_id]

    def remove(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.info("Removed session memory %s", session_id)

    def clear(self) -> None:
        with self._lock:
            self._store = {}
            logger.info("Cleared all session memories.")
