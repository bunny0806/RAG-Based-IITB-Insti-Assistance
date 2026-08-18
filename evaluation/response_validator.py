"""Validate generated responses and replace unsupported answers with a safe fallback."""

from __future__ import annotations

from typing import Sequence

from generation.generator import GeneratedAnswer
from retrieval.models import RetrievalResult
from utils.logging_utils import setup_logging
from observability import get_current_trace, get_metrics_collector, trace_stage

from .confidence import ConfidenceEstimator
from .groundedness import GroundednessChecker
from .models import EvaluationResult

logger = setup_logging("evaluation.log")


class ResponseValidator:
    """Evaluate generated answers for groundedness and confidence."""

    def __init__(
        self,
        groundedness_checker: GroundednessChecker | None = None,
        confidence_estimator: ConfidenceEstimator | None = None,
    ) -> None:
        self.groundedness_checker = groundedness_checker or GroundednessChecker()
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator()

    def validate(self, generated_answer: GeneratedAnswer, retrieval_results: Sequence[RetrievalResult]) -> EvaluationResult:
        """Validate an answer against the retrieved context and return an evaluation result."""
        with trace_stage("evaluation"):
            grounded, unsupported_claims = self.groundedness_checker.check(generated_answer.answer, retrieval_results)
            confidence_score = self.confidence_estimator.estimate(retrieval_results, grounded)

            if not grounded or confidence_score < 0.4:
                logger.warning("Answer marked as unsupported or low confidence.")
                safe_answer = "I don't know based on the available IIT Bombay documents."
                generated_answer.answer = safe_answer
                generated_answer.grounded = False
                unsupported_claims = unsupported_claims or ["low confidence"]
            else:
                logger.info("Answer passed groundedness and confidence checks.")

        retrieved_sources = [
            result.chunk.metadata.get("document_name")
            or result.chunk.metadata.get("filename")
            or result.chunk.metadata.get("title")
            or result.chunk.source
            for result in retrieval_results
            if isinstance(result.chunk.metadata, dict)
        ]

        result = EvaluationResult(
            grounded=grounded,
            confidence_score=confidence_score,
            reason="Answer is grounded in retrieved context." if grounded else "Answer lacks sufficient support from retrieved context.",
            retrieved_sources=retrieved_sources,
            unsupported_claims=list(unsupported_claims),
        )
        trace = get_current_trace()
        if trace is not None:
            get_metrics_collector().record(
                trace.trace_id,
                confidence_score=confidence_score,
                groundedness=grounded,
            )
        return result
