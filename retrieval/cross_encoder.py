"""Cross-encoder reranking support for retrieval candidates."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

from utils.logging_utils import setup_logging

logger = setup_logging("retrieval.log")


class CrossEncoderRanker:
    """Singleton wrapper for a sentence-transformers CrossEncoder model."""

    DEFAULT_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _instance: Optional["CrossEncoderRanker"] = None

    def __new__(cls, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> "CrossEncoderRanker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name
        self.device = device
        self._model = None
        self._initialized = True

    def _load_model(self):
        """Lazily load the CrossEncoder model on first inference."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            logger.error("sentence-transformers is required for cross-encoder reranking.")
            raise RuntimeError("sentence-transformers is required for cross-encoder reranking.") from exc

        logger.info("Loading CrossEncoder model '%s' on device '%s'", self.model_name, self.device)
        try:
            self._model = CrossEncoder(self.model_name, device=self.device)
        except Exception as exc:
            logger.error("Failed to load CrossEncoder model '%s': %s", self.model_name, exc)
            raise RuntimeError("Failed to initialize the CrossEncoder reranker.") from exc

        return self._model

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def score_pairs(
        self,
        query: str,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> List[float]:
        """Score query/document pairs using batch cross-encoder inference."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string for reranking.")
        if not texts:
            return []

        pairs = [(query, text) for text in texts]

        start_time = time.perf_counter()
        try:
            scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except Exception as exc:
            logger.error("CrossEncoder inference failed: %s", exc)
            raise RuntimeError("Cross-encoder scoring failed") from exc
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info("CrossEncoder scored %s pairs in %.2f ms", len(pairs), elapsed_ms)
        return [float(score) for score in scores]

    def reset_model(self) -> None:
        """Reset the loaded CrossEncoder model for testing or reconfiguration."""
        self._model = None
