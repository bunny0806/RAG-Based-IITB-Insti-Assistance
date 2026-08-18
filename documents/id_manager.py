"""Simple persistent ID allocator for vector IDs."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import List


class IDManager:
    """Allocate globally unique integer IDs and persist the next available ID.

    Stores a simple JSON file with {"next_id": <int>}.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._next_id = 1
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._next_id = int(data.get("next_id", 1))
        except Exception:
            self._next_id = 1

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump({"next_id": self._next_id}, handle)

    def allocate_ids(self, count: int) -> List[int]:
        """Allocate a contiguous block of `count` IDs and persist state."""
        if count <= 0:
            return []
        with self._lock:
            start = self._next_id
            ids = list(range(start, start + count))
            self._next_id = start + count
            self._save()
            return ids

    def peek_next(self) -> int:
        return self._next_id
