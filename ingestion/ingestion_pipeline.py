"""Pipeline for loading and validating multiple documents."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .base_loader import Document
from .loader_factory import LoaderFactory


class IngestionPipeline:
    """Coordinate ingestion across multiple sources."""

    def __init__(self, loader_factory: type[LoaderFactory] | None = None) -> None:
        self.loader_factory = loader_factory or LoaderFactory
        self.failed_sources: List[Tuple[str, str]] = []

    def ingest(self, sources: Sequence[str]) -> List[Document]:
        """Load one or more documents from file paths and URLs."""
        documents: List[Document] = []
        self.failed_sources = []

        for source in sources:
            try:
                loader = self.loader_factory.create_loader(source)
                document = loader.load()
                loader.validate(document)

                document.metadata.setdefault("loader_type", type(loader).__name__)
                document.metadata.setdefault("source_type", document.document_type)

                documents.append(document)
            except Exception as exc:  # pragma: no cover - defensive path
                self.failed_sources.append((source, str(exc)))

        return documents
