"""Data models for the preprocessing layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class Chunk:
    """A cleaned and segmented chunk of a document."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: str = ""
    chunk_index: int = 0
    source: str = ""
    document_type: str = "unknown"
    start_char: int = 0
    end_char: int = 0
