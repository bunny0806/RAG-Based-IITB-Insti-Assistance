"""Reciprocal Rank Fusion utilities for hybrid retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from retrieval.models import RetrievalResult
from utils.logging_utils import setup_logging

logger = setup_logging("retrieval.log")


def reciprocal_rank_fusion(
    ranked_results: Sequence[Sequence[RetrievalResult]],
    k: int = 60,
    top_k: int = 5,
) -> List[RetrievalResult]:
    """Fuse ranked retrieval lists using reciprocal rank fusion."""
    if k <= 0:
        raise ValueError("rrf_k must be positive.")
    if top_k <= 0:
        raise ValueError("final_top_k must be positive.")

    if not ranked_results:
        return []

    score_map: Dict[str, float] = defaultdict(float)
    best_rank: Dict[str, int] = {}
    representative: Dict[str, RetrievalResult] = {}

    for ranked_list in ranked_results:
        for rank, result in enumerate(ranked_list, start=1):
            key = _result_key(result)
            score_map[key] += 1.0 / (k + rank)
            if key not in representative or rank < best_rank[key]:
                representative[key] = result
                best_rank[key] = rank

    fused_results = sorted(
        representative.values(),
        key=lambda result: (-score_map[_result_key(result)], best_rank[_result_key(result)]),
    )

    final_results: List[RetrievalResult] = []
    for rank, result in enumerate(fused_results[:top_k], start=1):
        fused_result = RetrievalResult(
            chunk=result.chunk,
            score=score_map[_result_key(result)],
            rank=rank,
            retrieval_method="hybrid",
            metadata=result.metadata,
        )
        final_results.append(fused_result)

    logger.info(
        "Reciprocal Rank Fusion produced %s final results from %s sources.",
        len(final_results),
        len(ranked_results),
    )
    return final_results


def _result_key(result: RetrievalResult) -> str:
    """Create a deterministic key for a retrieval result to identify duplicates."""
    chunk_id = getattr(result.chunk, "chunk_id", None)
    if chunk_id:
        return str(chunk_id)

    metadata = getattr(result.chunk, "metadata", {}) or {}
    source = metadata.get("document_name") or metadata.get("filename") or getattr(result.chunk, "source", "")
    text_hash = hash(getattr(result.chunk, "text", ""))
    return f"{source}-{text_hash}"
