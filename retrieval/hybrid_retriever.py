"""Hybrid retrieval combining dense and BM25 reranking."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

from retrieval.bm25_retriever import BM25Retriever
from retrieval.models import RetrievalResult
from retrieval.rank_fusion import reciprocal_rank_fusion
from retrieval.retriever import Retriever
from utils.logging_utils import setup_logging

logger = setup_logging("retrieval.log")


class HybridRetriever:
    """Combine dense and sparse retrieval results using reciprocal rank fusion."""

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: BM25Retriever,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Retrieve and fuse results from dense and BM25 search sources."""
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string.")
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("query_embedding must be a numpy array.")

        dense_results: List[RetrievalResult] = []
        bm25_results: List[RetrievalResult] = []

        start_time = time.perf_counter()
        dense_start = time.perf_counter()
        try:
            dense_results = self.dense_retriever.search(query_embedding, top_k=self.dense_top_k)
            logger.info("Dense retrieval returned %s results.", len(dense_results))
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Dense retrieval failed: %s", exc)
        dense_latency_ms = (time.perf_counter() - dense_start) * 1000

        bm25_start = time.perf_counter()
        try:
            bm25_results = self.bm25_retriever.search(query_text, top_k=self.bm25_top_k)
            logger.info("BM25 retrieval returned %s results.", len(bm25_results))
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("BM25 retrieval failed: %s", exc)
        bm25_latency_ms = (time.perf_counter() - bm25_start) * 1000

        fusion_start = time.perf_counter()
        if not dense_results and not bm25_results:
            raise RuntimeError("Hybrid retrieval failed to retrieve any results from both dense and BM25 sources.")

        fused_results = reciprocal_rank_fusion(
            [source for source in [dense_results, bm25_results] if source],
            k=self.rrf_k,
            top_k=top_k or self.final_top_k,
        )
        fusion_latency_ms = (time.perf_counter() - fusion_start) * 1000
        total_latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Hybrid retrieval latencies: dense=%.2fms bm25=%.2fms fusion=%.2fms total=%.2fms",
            dense_latency_ms,
            bm25_latency_ms,
            fusion_latency_ms,
            total_latency_ms,
        )
        logger.info(
            "Hybrid retrieval final chunks: %s",
            [result.chunk.chunk_id for result in fused_results],
        )

        return fused_results
