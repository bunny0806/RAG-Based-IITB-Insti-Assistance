"""BM25-based sparse retrieval for chunk-level search."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rank_bm25 import BM25Okapi

from preprocessing.models import Chunk
from retrieval.models import RetrievalResult
from utils.logging_utils import setup_logging

logger = setup_logging("retrieval.log")
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 indexing and query matching."""
    return [token.lower() for token in _TOKEN_RE.findall(text) if token]


class BM25Retriever:
    """Sparse BM25 retriever that supports incremental indexing and persistence."""

    def __init__(self, index_path: str | Path, top_k: int = 10) -> None:
        self.index_path = Path(index_path)
        self.top_k = top_k
        self._bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self.documents)

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Build a BM25 index from a sequence of chunks."""
        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty chunk list.")

        self.documents = [chunk.text for chunk in chunks]
        self.metadata = [self._chunk_to_metadata(chunk) for chunk in chunks]
        self._refresh_model()
        self.save()

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Append new chunks to the BM25 index and refresh the model."""
        if not chunks:
            return

        self.documents.extend(chunk.text for chunk in chunks)
        self.metadata.extend(self._chunk_to_metadata(chunk) for chunk in chunks)
        self._refresh_model()
        self.save()

    def save(self) -> None:
        """Persist BM25 documents and metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self.documents,
            "metadata": self.metadata,
        }
        with self.index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Saved BM25 index to %s", self.index_path)

    def load(self) -> None:
        """Load BM25 documents and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {self.index_path}")

        with self.index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.documents = [str(doc) for doc in payload.get("documents", [])]
        self.metadata = [dict(item) for item in payload.get("metadata", [])]

        if not self.documents:
            raise ValueError("Loaded BM25 index contains no documents.")

        self._refresh_model()
        logger.info("Loaded BM25 index from %s", self.index_path)

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Return the top-k BM25 retrieval results for a text query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not self.is_ready:
            raise RuntimeError("BM25 index is not loaded or initialized.")

        effective_top_k = top_k or self.top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be positive.")

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        tokenized_documents = [_tokenize(text) for text in self.documents]
        document_token_sets = [set(tokens) for tokens in tokenized_documents]
        match_strength = [
            sum(1 for token in tokenized_query if token in document_tokens)
            for document_tokens in document_token_sets
        ]

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: (-scores[idx], -match_strength[idx], idx),
        )

        results: List[RetrievalResult] = []
        for rank, document_index in enumerate(ranked_indices[: min(effective_top_k, len(ranked_indices))], start=1):
            score = float(scores[document_index])
            metadata = self.metadata[document_index]
            chunk = self._build_chunk_from_metadata(metadata)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_method="bm25",
                    metadata=metadata,
                )
            )

        logger.info("BM25 retrieved %s result(s) for query.", len(results))
        return results

    def _refresh_model(self) -> None:
        """Rebuild the BM25 model from current documents."""
        tokenized_documents = [_tokenize(text) for text in self.documents]
        self._bm25 = BM25Okapi(tokenized_documents)

    def _chunk_to_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert a chunk object into a storable metadata dictionary."""
        metadata = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "source": chunk.source,
            "document_type": chunk.document_type,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            **chunk.metadata,
        }
        return metadata

    def _build_chunk_from_metadata(self, metadata: Dict[str, Any]) -> Chunk:
        """Create a Chunk object from stored metadata."""
        return Chunk(
            chunk_id=str(metadata.get("chunk_id", "")),
            text=str(metadata.get("text", "")),
            metadata={key: value for key, value in metadata.items() if key not in {"text", "chunk_id", "document_id", "chunk_index", "source", "document_type", "start_char", "end_char"}},
            document_id=str(metadata.get("document_id", "")),
            chunk_index=int(metadata.get("chunk_index", 0)),
            source=str(metadata.get("source", "")),
            document_type=str(metadata.get("document_type", "unknown")),
            start_char=int(metadata.get("start_char", 0)),
            end_char=int(metadata.get("end_char", 0)),
        )
