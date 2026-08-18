"""End-to-end preprocessing pipeline for converting Documents into chunks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ingestion.base_loader import Document

from .cleaner import TextCleaner
from .chunker import Chunker
from .models import Chunk
from utils.logging_utils import setup_logging

logger = setup_logging("preprocessing.log")


class ChunkPipeline:
    """Clean documents and produce chunk objects for downstream stages."""

    def __init__(
        self,
        cleaner: Optional[TextCleaner] = None,
        chunker: Optional[Chunker] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        """Process a sequence of documents into chunk objects."""
        if not documents:
            logger.warning("No documents provided to chunk pipeline.")
            return []

        chunks: List[Chunk] = []
        for document in documents:
            try:
                cleaned_text = self.cleaner.clean(document.content)
                logger.info("Document cleaned: %s", document.source)

                document_id = self._build_document_id(document)
                chunk_objects = self.chunker.build_chunks(
                    text=cleaned_text,
                    document_id=document_id,
                    source=document.source,
                    document_type=document.document_type,
                    metadata=self._build_document_metadata(document),
                )
                logger.info("Document chunked: %s -> %s chunk(s)", document.source, len(chunk_objects))
                chunks.extend(chunk_objects)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.error("Failed to preprocess document %s: %s", document.source, exc)
                raise RuntimeError(f"Unable to preprocess document: {document.source}") from exc

        return chunks

    def _build_document_id(self, document: Document) -> str:
        """Create a deterministic document ID from the source and type."""
        source_name = document.source.replace("/", "_").replace(".", "_").replace(":", "_")
        return f"{document.document_type}_{source_name}"

    def _build_document_metadata(self, document: Document) -> Dict[str, Any]:
        """Attach ingestion metadata to chunk metadata."""
        metadata = dict(document.metadata)
        metadata.setdefault("source", document.source)
        metadata.setdefault("document_type", document.document_type)
        return metadata
