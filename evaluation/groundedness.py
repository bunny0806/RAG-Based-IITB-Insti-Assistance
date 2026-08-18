"""Heuristic groundedness checks for generated answers."""

from __future__ import annotations

import re
from typing import List, Sequence

from retrieval.models import RetrievalResult
from utils.logging_utils import setup_logging

logger = setup_logging("evaluation.log")


class GroundednessChecker:
    """Check whether a generated answer is supported by retrieved context."""

    def check(self, answer: str, retrieval_results: Sequence[RetrievalResult]) -> tuple[bool, List[str]]:
        """Return groundedness status and unsupported claims using lightweight heuristics."""
        if not isinstance(answer, str) or not answer.strip():
            return False, ["empty answer"]

        if not retrieval_results:
            logger.warning("No retrieval results available for groundedness check.")
            return False, ["no retrieved context"]

        normalized_answer = self._normalize_text(answer)
        supported_claims = 0
        unsupported_claims: List[str] = []

        context_text = self._collect_context_text(retrieval_results)
        if not context_text:
            return False, ["retrieved context empty"]

        sentences = self._split_sentences(normalized_answer)
        if not sentences:
            return False, ["no answer sentences to evaluate"]

        for sentence in sentences:
            if self._is_supported(sentence, context_text):
                supported_claims += 1
            else:
                unsupported_claims.append(sentence)

        grounded = len(unsupported_claims) == 0 or (supported_claims / max(1, len(sentences)) >= 0.5)
        if grounded:
            logger.info("Answer appears grounded based on retrieved context.")
        else:
            logger.warning("Answer contains unsupported claims: %s", unsupported_claims)
        return grounded, unsupported_claims

    def _normalize_text(self, text: str) -> str:
        """Normalize text for lightweight matching."""
        normalized = re.sub(r"\s+", " ", text.strip())
        return normalized.lower()

    def _collect_context_text(self, retrieval_results: Sequence[RetrievalResult]) -> str:
        """Concatenate retrieved chunk text into a single context body."""
        chunks = [result.chunk.text for result in retrieval_results if getattr(result.chunk, "text", "")]
        return "\n".join(chunks)

    def _split_sentences(self, text: str) -> List[str]:
        """Split answer text into simple sentence-like units."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _is_supported(self, sentence: str, context_text: str) -> bool:
        """Heuristically check if a sentence is supported by the context."""
        if not sentence:
            return False
        context = self._normalize_text(context_text)
        if sentence in context:
            return True
        return len(set(sentence.split()) & set(context.split())) >= 3
