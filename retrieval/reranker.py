"""Second-stage reranking for retrieval results using a CrossEncoder."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

from retrieval.cross_encoder import CrossEncoderRanker
from retrieval.models import RetrievalResult
from utils.logging_utils import setup_logging
from observability import get_current_trace, get_metrics_collector, trace_stage

logger = setup_logging("retrieval.log")


class Reranker:
    """Re-rank retrieval candidates using a CrossEncoder model."""

    def __init__(
        self,
        cross_encoder_model: str = CrossEncoderRanker.DEFAULT_MODEL,
        candidate_k: int = 20,
        reranked_k: int = 5,
        device: str = "cpu",
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")
        if reranked_k <= 0:
            raise ValueError("reranked_k must be positive.")

        self.cross_encoder_model = cross_encoder_model
        self.candidate_k = candidate_k
        self.reranked_k = reranked_k
        self.device = device
        self._ranker = CrossEncoderRanker(model_name=cross_encoder_model, device=device)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Re-rank candidate RetrievalResult objects and return the top-k final results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string for reranking.")
        if candidates is None:
            raise ValueError("Candidates must be provided for reranking.")

        final_top_k = top_k or self.reranked_k
        if final_top_k <= 0:
            raise ValueError("top_k must be positive.")

        candidate_list = list(candidates)[: self.candidate_k]
        candidate_count = len(candidate_list)
        if candidate_count == 0:
            logger.info("Reranker received zero candidates, returning empty result set.")
            return []

        texts = [candidate.chunk.text for candidate in candidate_list]

        start_time = time.perf_counter()
        with trace_stage("reranking"):
            scores = self._ranker.score_pairs(query, texts)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        reranked = [
            (score, index, candidate)
            for index, (score, candidate) in enumerate(zip(scores, candidate_list))
        ]
        reranked.sort(key=lambda item: (-item[0], item[1]))

        final_results: List[RetrievalResult] = []
        for rank, (score, _, candidate) in enumerate(reranked[:final_top_k], start=1):
            final_results.append(
                RetrievalResult(
                    chunk=candidate.chunk,
                    score=score,
                    rank=rank,
                    retrieval_method="cross_encoder",
                    metadata=candidate.metadata,
                )
            )

        logger.info(
            "Reranker evaluated %s candidates and returned %s final results in %.2f ms.",
            candidate_count,
            len(final_results),
            elapsed_ms,
        )
        logger.info("Reranker top scores: %s", [result.score for result in final_results])
        logger.info("Reranker selected chunks: %s", [result.chunk.chunk_id for result in final_results])
        trace = get_current_trace()
        if trace is not None:
            get_metrics_collector().record(
                trace.trace_id,
                reranker_score=float(final_results[0].score) if final_results else 0.0,
            )

        return final_results
