"""Data models for the evaluation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class EvaluationResult:
    """Represents the trustworthiness assessment of a generated answer."""

    grounded: bool
    confidence_score: float
    reason: str
    retrieved_sources: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
