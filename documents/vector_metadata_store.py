"""Store mapping from vector_id -> chunk metadata persistently."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Any


class VectorMetadataStore:
    """Persistent mapping of vector_id to chunk-level metadata.

    Stored on disk as a JSON object mapping stringified vector IDs to metadata dicts.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._map: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._map = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._map = {k: v for k, v in data.items()}
        except Exception:
            self._map = {}

    def _save(self) -> None:
        with self._lock:
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(self._map, handle, indent=2)

    def add_entries(self, entries: Dict[int, Dict[str, Any]]) -> None:
        """Add mapping entries where keys are integer vector IDs."""
        with self._lock:
            for vid, meta in entries.items():
                self._map[str(int(vid))] = meta
            self._save()

    def remove_ids(self, ids: List[int]) -> None:
        with self._lock:
            for vid in ids:
                self._map.pop(str(int(vid)), None)
            self._save()

    def get_ids_for_document(self, document_id: str) -> List[int]:
        return [int(k) for k, v in self._map.items() if v.get("document_id") == document_id]

    def get_all(self) -> Dict[int, Dict[str, Any]]:
        return {int(k): v for k, v in self._map.items()}

    def exists(self, vector_id: int) -> bool:
        return str(int(vector_id)) in self._map
