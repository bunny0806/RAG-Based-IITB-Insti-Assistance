"""End-to-end retrieval pipeline for user queries."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

from embeddings.embedder import Embedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from utils.logging_utils import setup_logging
from observability import get_current_trace, get_metrics_collector, trace_stage

from .models import RetrievalResult
from .query_processor import QueryProcessor
from .retriever import Retriever
from config import (
    CROSS_ENCODER_MODEL,
    RERANKER_CANDIDATE_K,
    RERANKER_FINAL_K,
    RERANKING_ENABLED,
)

logger = setup_logging("retrieval.log")


class RetrievalPipeline:
    """Coordinate query processing, embedding generation, and retrieval."""

    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        embedder: Optional[Embedder] = None,
        retriever: Optional[Retriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        retrieval_mode: str = "dense",
    ) -> None:
        self.query_processor = query_processor or QueryProcessor()
        self.embedder = embedder or Embedder()
        self.retriever = retriever
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or (Reranker(cross_encoder_model=CROSS_ENCODER_MODEL) if RERANKING_ENABLED else None)
        self.retrieval_mode = retrieval_mode

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Process a query and retrieve ranked results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        logger.info("Incoming query: %s", query)
        with trace_stage("query_processing"):
            processed_query = self.query_processor.process(query)
        logger.info("Processed query: %s", processed_query)

        start_time = time.perf_counter()
        with trace_stage("retrieval"):
            try:
                query_embedding = self.embedder.encode_single(processed_query)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.error("Embedding generation failed: %s", exc)
                raise RuntimeError("Failed to generate embedding for query") from exc

            if self.retrieval_mode == "dense":
                if self.retriever is None:
                    raise RuntimeError("Dense retriever has not been configured.")
                try:
                    results = self.retriever.search(query_embedding, top_k=top_k)
                except Exception as exc:  # pragma: no cover - defensive path
                    logger.error("Dense retrieval failed: %s", exc)
                    raise RuntimeError("Failed to retrieve chunks using dense retrieval") from exc
            elif self.retrieval_mode == "bm25":
                if self.bm25_retriever is None:
                    raise RuntimeError("BM25 retriever has not been configured.")
                try:
                    results = self.bm25_retriever.search(processed_query, top_k=top_k)
                except Exception as exc:  # pragma: no cover - defensive path
                    logger.error("BM25 retrieval failed: %s", exc)
                    raise RuntimeError("Failed to retrieve chunks using BM25") from exc
            elif self.retrieval_mode == "hybrid":
                if self.hybrid_retriever is None:
                    raise RuntimeError("Hybrid retriever has not been configured.")
                try:
                    results = self.hybrid_retriever.search(processed_query, query_embedding, top_k=top_k)
                except Exception as exc:  # pragma: no cover - defensive path
                    logger.error("Hybrid retrieval failed: %s", exc)
                    raise RuntimeError("Failed to retrieve chunks using hybrid retrieval") from exc
            else:
                raise ValueError(
                    f"Invalid retrieval_mode: {self.retrieval_mode}. "
                    "Expected one of: dense, bm25, hybrid."
                )

        if RERANKING_ENABLED and self.reranker is not None and results:
            try:
                logger.info("Starting reranking with %s candidates.", len(results))
                reranked_results = self.reranker.rerank(processed_query, results, top_k=top_k)
                results = reranked_results
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Reranking failed, returning original results: %s", exc)


        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Retrieval completed in %.2f ms", elapsed_ms)
        logger.info("Retrieved %s chunk(s)", len(results))
        trace = get_current_trace()
        if trace is not None:
            scores = [float(result.score) for result in results]
            get_metrics_collector().record(
                trace.trace_id,
                number_of_chunks=len(results),
                average_similarity_score=(sum(scores) / len(scores)) if scores else 0.0,
            )
        return results
