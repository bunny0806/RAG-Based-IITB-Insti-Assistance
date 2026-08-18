"""Intelligent query processing pipeline for RAG retrieval."""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Optional

from config import (
    CLASSIFICATION_ENABLED,
    QUERY_EXPANSION_ENABLED,
    QUERY_REWRITING_ENABLED,
)
from query.classifier import QueryClassifier
from query.expander import QueryExpander
from query.rewriter import QueryRewriter
from utils.logging_utils import setup_logging

logger = setup_logging("query.log")


@dataclasses.dataclass(frozen=True)
class ProcessedQuery:
    """Structured result of query processing."""

    original_query: str
    expanded_query: str
    rewritten_query: str
    category: str
    final_query: str


class QueryPipeline:
    """Run expansion, classification, and rewriting before retrieval."""

    def __init__(
        self,
        expander: Optional[QueryExpander] = None,
        classifier: Optional[QueryClassifier] = None,
        rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self.expander = expander or QueryExpander()
        self.classifier = classifier or QueryClassifier()
        self.rewriter = rewriter or QueryRewriter()

    def process(self, query: str) -> ProcessedQuery:
        """Process the incoming query through expansion, classification, and rewriting."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        logger.info("Original query: %s", query)
        start_time = time.perf_counter()

        expanded_query = self.expander.expand(query) if self._is_feature_enabled("QUERY_EXPANSION_ENABLED", QUERY_EXPANSION_ENABLED) else query.strip()
        category = self.classifier.classify(expanded_query) if self._is_feature_enabled("CLASSIFICATION_ENABLED", CLASSIFICATION_ENABLED) else "General"
        rewritten_query = self.rewriter.rewrite(expanded_query) if self._is_feature_enabled("QUERY_REWRITING_ENABLED", QUERY_REWRITING_ENABLED) else expanded_query
        final_query = rewritten_query

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Query pipeline complete: original='%s' expanded='%s' rewritten='%s' category='%s' final='%s' latency=%.2fms",
            query,
            expanded_query,
            rewritten_query,
            category,
            final_query,
            elapsed_ms,
        )

        return ProcessedQuery(
            original_query=query,
            expanded_query=expanded_query,
            rewritten_query=rewritten_query,
            category=category,
            final_query=final_query,
        )

    @staticmethod
    def _is_feature_enabled(env_name: str, default_value: bool) -> bool:
        value = os.getenv(env_name)
        if value is None:
            return default_value
        return value.strip().lower() in {"1", "true", "yes", "on"}
