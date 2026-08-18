"""Retrieval logic for ranking chunks from a vector store."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from preprocessing.models import Chunk
from utils.logging_utils import setup_logging

from .models import RetrievalResult

logger = setup_logging("retrieval.log")


class VectorStoreProtocol(Protocol):
    """Protocol that describes the minimal vector store interface required by Retriever."""

    index: Any
    metadata: List[Dict[str, Any]]

    def stats(self) -> Dict[str, Any]:
        """Return store statistics."""


class Retriever:
    """Rank chunks by cosine similarity against a query embedding."""

    def __init__(self, vector_store: VectorStoreProtocol, top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Return the top-k retrieval results for the given query embedding."""
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("Query embedding must be a numpy array.")

        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be a 1D array.")

        if self.vector_store.index is None:
            raise ValueError("Vector store index has not been initialized.")

        effective_top_k = top_k or self.top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be positive.")

        normalized_query = self._normalize_vector(query_embedding)
        index = self.vector_store.index

        if index.ntotal == 0:
            logger.warning("Vector store contains no vectors.")
            return []

        if index.d != normalized_query.shape[0]:
            raise ValueError(
                f"Embedding dimension mismatch: expected {index.d}, got {normalized_query.shape[0]}"
            )

        try:
            distances, indices = index.search(
                normalized_query.reshape(1, -1).astype(np.float32),
                min(effective_top_k, index.ntotal),
            )
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error("FAISS search failed: %s", exc)
            raise RuntimeError("Failed to retrieve results from vector store") from exc

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(indices[0], start=1):
            if idx < 0:
                continue
            metadata = self._get_metadata_for_index(idx)
            if not metadata:
                continue

            chunk = self._build_chunk_from_metadata(metadata)
            score = float(distances[0][rank - 1])
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_method="faiss_cosine",
                    metadata=metadata,
                )
            )

        logger.info("Retrieved %s chunk(s) with scores %s.", len(results), [result.score for result in results])
        return results

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalize the query vector for cosine similarity compatibility."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Query embedding must not be zero vector.")
        return vector.astype(np.float32) / norm

    def _get_metadata_for_index(self, index: int) -> Dict[str, Any]:
        """Return metadata for a vector label (supports legacy positional indices and ID labels)."""
        # If the vector_store exposes a direct lookup by vector id, prefer it.
        try:
            # treat `index` as a faiss returned label (vector id)
            if hasattr(self.vector_store, "get_metadata_by_vector_id"):
                meta = self.vector_store.get_metadata_by_vector_id(int(index))
                return meta or {}
        except Exception:
            pass

        # Fallback to legacy positional metadata list
        if index < 0 or index >= len(getattr(self.vector_store, "metadata", [])):
            return {}
        metadata = self.vector_store.metadata[index]
        if not isinstance(metadata, dict):
            return {}
        return metadata

    def _build_chunk_from_metadata(self, metadata: Dict[str, Any]) -> Chunk:
        """Create a Chunk object from metadata if possible."""
        chunk = Chunk(
            chunk_id=str(metadata.get("chunk_id", "")),
            text=str(metadata.get("text", "")),
            metadata={key: value for key, value in metadata.items() if key != "text"},
            document_id=str(metadata.get("document_id", "")),
            chunk_index=int(metadata.get("chunk_index", 0)),
            source=str(metadata.get("source", "")),
            document_type=str(metadata.get("document_type", "unknown")),
            start_char=int(metadata.get("start_char", 0)),
            end_char=int(metadata.get("end_char", 0)),
        )
        return chunk
