"""Pipeline for embedding chunks and maintaining chunk-to-embedding mappings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from preprocessing.models import Chunk
from utils.logging_utils import setup_logging

from .embedder import Embedder

logger = setup_logging("embeddings.log")


class EmbeddingPipeline:
    """Generate embeddings for preprocessing chunks and preserve mapping metadata."""

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or Embedder()

    def embed_chunks(self, chunks: Sequence[Chunk]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Embed a sequence of chunks and return embeddings plus chunk metadata."""
        if not chunks:
            logger.warning("No chunks provided to embedding pipeline.")
            return np.empty((0, self.embedder.embedding_dimension), dtype=np.float32), []

        texts = [chunk.text for chunk in chunks]
        try:
            embeddings = self.embedder.encode_many_with_progress(texts)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("Embedding generation failed: %s", exc)
            raise RuntimeError("Failed to generate embeddings for chunks") from exc

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        self._validate_embeddings(embeddings, len(chunks))

        metadata_list = []
        for index, chunk in enumerate(chunks):
            metadata = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "source": chunk.source,
                "document_type": chunk.document_type,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata,
            }
            metadata_list.append(metadata)

        logger.info("Generated %s embeddings with dimension %s.", len(chunks), embeddings.shape[1])
        return embeddings, metadata_list

    def _validate_embeddings(self, embeddings: np.ndarray, expected_count: int) -> None:
        """Ensure the produced embeddings match the expected count and shape."""
        if embeddings.shape[0] != expected_count:
            raise ValueError(
                f"Embedding count mismatch: expected {expected_count}, got {embeddings.shape[0]}"
            )
        if embeddings.shape[1] <= 0:
            raise ValueError("Embedding dimension must be positive.")
        logger.info("Embedding validation passed for %s vectors.", expected_count)
