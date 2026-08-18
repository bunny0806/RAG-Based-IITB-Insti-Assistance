"""Shared abstractions and the reusable Document model for ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from utils.logging_utils import setup_logging

logger = setup_logging("ingestion.log")


@dataclass(slots=True)
class Document:
    """Standardized representation of a loaded document."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    document_type: str = "unknown"


class BaseLoader(ABC):
    """Abstract base class for all ingestion loaders."""

    def __init__(self, source: str, document_type: str) -> None:
        self.source = source
        self.document_type = document_type
        self.logger = logger

    @abstractmethod
    def load(self) -> Document:
        """Load the document and return a standardized Document object."""

    @abstractmethod
    def validate(self, document: Document) -> None:
        """Validate that the loaded document satisfies minimum requirements."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata collected by the loader."""
