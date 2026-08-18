"""Factory class for creating the right loader for a source."""

from __future__ import annotations

from typing import Dict, Type
from urllib.parse import urlparse

from .base_loader import BaseLoader
from .pdf_loader import PDFLoader
from .web_loader import WebLoader


class LoaderFactory:
    """Select an appropriate loader based on source type."""

    registry: Dict[str, Type[BaseLoader]] = {
        "pdf": PDFLoader,
        "web": WebLoader,
    }

    @classmethod
    def create_loader(cls, source: str) -> BaseLoader:
        """Create a loader for a local file path or web URL."""
        if not source or not source.strip():
            raise ValueError("Source cannot be empty.")

        normalized_source = source.strip()
        if cls._is_url(normalized_source):
            return cls.registry["web"](normalized_source)
        if normalized_source.lower().endswith(".pdf"):
            return cls.registry["pdf"](normalized_source)

        raise ValueError(f"Unsupported source type: {source}")

    @classmethod
    def register_loader(cls, name: str, loader_class: Type[BaseLoader]) -> None:
        """Register a new loader implementation for future expansion."""
        cls.registry[name.lower()] = loader_class

    @staticmethod
    def _is_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
