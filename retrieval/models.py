"""Data models for the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from preprocessing.models import Chunk


@dataclass(slots=True)
class RetrievalResult:
    """A ranked retrieval result returned by the retrieval layer."""

    chunk: Chunk
    score: float
    rank: int
    retrieval_method: str = "faiss_cosine"
    metadata: Dict[str, Any] = field(default_factory=dict)
