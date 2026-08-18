"""Utilities for creating, updating, rebuilding, and loading vector indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings.embedding_pipeline import EmbeddingPipeline
from preprocessing.models import Chunk
from utils.logging_utils import setup_logging

from .faiss_store import FAISSStore

logger = setup_logging("vectorstore.log")


class IndexManager:
    """Build and manage FAISS indexes and metadata from chunks."""

    def __init__(self, index_path: str | Path, metadata_path: Optional[str | Path] = None) -> None:
        self.store = FAISSStore(index_path=index_path, metadata_path=metadata_path)
        self.embedding_pipeline = EmbeddingPipeline()

    def build_new_index(self, chunks: Sequence[Chunk]) -> FAISSStore:
        """Create a new index from chunks."""
        if not chunks:
            raise ValueError("Cannot build an index from empty chunks.")

        embeddings, metadata = self.embedding_pipeline.embed_chunks(chunks)
        self.store.create_index(embeddings.shape[1])
        self.store.add_embeddings(embeddings, metadata)
        self.store.save()
        logger.info("Built new index with %s vectors.", len(chunks))
        return self.store

    def append_to_index(self, chunks: Sequence[Chunk]) -> FAISSStore:
        """Append embeddings for new chunks to an existing index."""
        if self.store.index is None:
            raise ValueError("No existing index available to append to.")

        embeddings, metadata = self.embedding_pipeline.embed_chunks(chunks)
        self.store.add_embeddings(embeddings, metadata)
        self.store.save()
        logger.info("Appended %s vectors to existing index.", len(chunks))
        return self.store

    def rebuild_index(self, chunks: Sequence[Chunk]) -> FAISSStore:
        """Rebuild the index from scratch using provided chunks."""
        logger.info("Rebuilding index from %s chunks.", len(chunks))
        return self.build_new_index(chunks)

    def load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata from disk."""
        self.store.load()
        return self.store.metadata
