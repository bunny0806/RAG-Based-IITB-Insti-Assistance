"""Sentence-transformer based embedding utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.logging_utils import setup_logging

logger = setup_logging("embeddings.log")


class Embedder:
    """Lazy-loading sentence-transformer embedder with batching support."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension after the model is loaded."""
        if self._embedding_dim is None:
            self.load_model()
        assert self._embedding_dim is not None
        return self._embedding_dim

    def load_model(self) -> SentenceTransformer:
        """Load the sentence-transformer model once and cache it."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding model loaded. Dimension: %s", self._embedding_dim)
        return self._model

    def encode_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedder requires non-empty text.")
        embedding = self.load_model().encode([text], convert_to_numpy=True)[0]
        return np.asarray(embedding, dtype=np.float32)

    def encode_many(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings in batches."""
        if not texts:
            raise ValueError("No texts provided for embedding.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts must be non-empty strings.")

        model = self.load_model()
        logger.info("Generating embeddings for %s text(s).", len(texts))
        embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_many_with_progress(self, texts: List[str]) -> np.ndarray:
        """Embed texts with a progress bar for larger batches."""
        if not texts:
            raise ValueError("No texts provided for embedding.")

        model = self.load_model()
        logger.info("Generating embeddings with progress for %s text(s).", len(texts))
        batch_size = 32
        all_embeddings: List[np.ndarray] = []

        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[start : start + batch_size]
            batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(np.asarray(batch_embeddings, dtype=np.float32))

        return np.vstack(all_embeddings)
