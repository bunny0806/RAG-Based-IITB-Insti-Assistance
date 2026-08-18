"""Estimate confidence for generated answers using retrieval signals."""

from __future__ import annotations

from typing import Sequence

from retrieval.models import RetrievalResult
from utils.logging_utils import setup_logging

logger = setup_logging("evaluation.log")


class ConfidenceEstimator:
    """Estimate confidence from retrieval quality and grounding signals."""

    def estimate(
        self,
        retrieval_results: Sequence[RetrievalResult],
        grounded: bool,
    ) -> float:
        """Return a confidence score between 0.0 and 1.0."""
        if not retrieval_results:
            logger.warning("Confidence estimate set to 0.0 because no retrieval results were provided.")
            return 0.0

        scores = [max(float(result.score), 0.0) for result in retrieval_results]
        if not scores:
            return 0.0

        avg_score = sum(scores) / len(scores)
        chunk_count_factor = min(len(retrieval_results) / 5.0, 1.0)
        similarity_factor = min(avg_score / 10.0, 1.0)
        grounded_factor = 1.0 if grounded else 0.0

        confidence = 0.4 * chunk_count_factor + 0.3 * similarity_factor + 0.3 * grounded_factor
        confidence = max(0.0, min(1.0, confidence))
        logger.info("Estimated confidence: %.3f", confidence)
        return confidence
