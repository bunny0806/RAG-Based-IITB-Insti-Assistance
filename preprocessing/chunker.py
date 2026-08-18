"""Chunking logic based on LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Chunk
from utils.logging_utils import setup_logging

logger = setup_logging("preprocessing.log")


class Chunker:
    """Split cleaned text into semantically meaningful chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def split_text(self, text: str) -> List[str]:
        """Split a cleaned string into multiple chunks."""
        if not isinstance(text, str):
            raise TypeError("Chunker expects a string input.")
        if not text.strip():
            raise ValueError("Chunker cannot split empty text.")

        chunks = self._splitter.split_text(text)
        logger.info("Document chunked into %s chunk(s).", len(chunks))
        for index, chunk in enumerate(chunks):
            logger.info("Chunk %s size: %s characters", index + 1, len(chunk))
        return chunks

    def build_chunks(
        self,
        text: str,
        document_id: str,
        source: str,
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Create Chunk objects from a text body and document metadata."""
        chunks = self.split_text(text)
        document_metadata = metadata or {}
        built_chunks: List[Chunk] = []

        for index, chunk_text in enumerate(chunks):
            start_char = text.find(chunk_text, 0)
            if start_char == -1:
                start_char = index * self.chunk_size
            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                chunk_id=self._build_chunk_id(document_id, index),
                text=chunk_text,
                metadata={
                    **document_metadata,
                    "chunk_index": index,
                    "chunk_size": len(chunk_text),
                    "document_id": document_id,
                    "source": source,
                    "document_type": document_type,
                },
                document_id=document_id,
                chunk_index=index,
                source=source,
                document_type=document_type,
                start_char=start_char,
                end_char=end_char,
            )
            built_chunks.append(chunk)

        return built_chunks

    def _build_chunk_id(self, document_id: str, index: int) -> str:
        """Create deterministic chunk IDs based on document ID and chunk index."""
        return f"{document_id}_chunk_{index:03d}"
