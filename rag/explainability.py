"""Explainability utilities for RAG retrieval and answer transparency."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from retrieval.models import RetrievalResult


@dataclass(slots=True)
class SourceCitation:
    source_name: str
    page_number: str
    retrieval_method: str
    score: float
    overlap_count: int
    snippet: str


@dataclass(slots=True)
class ExplainabilityReport:
    citations: List[SourceCitation] = field(default_factory=list)
    summary: str = ""
    query_alignment: str = ""
    # memory-aware fields
    resolved_query: str = ""
    summary_used: str = ""
    memory_hits: int = 0


class ExplainabilityBuilder:
    """Build a lightweight explanation report from retrieved context."""

    def build(self, query: str, retrieval_results: Sequence[RetrievalResult], resolved_query: str | None = None, summary_used: str | None = None, memory_hits: int = 0) -> ExplainabilityReport:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string for explanation generation.")

        citations: List[SourceCitation] = []
        query_terms = self._normalize_terms(query)

        for result in retrieval_results:
            chunk = result.chunk
            metadata = getattr(chunk, "metadata", {}) or {}
            source_name = (
                metadata.get("document_name")
                or metadata.get("filename")
                or metadata.get("title")
                or chunk.source
                or "Unknown source"
            )
            page_number = str(metadata.get("page_number", "n/a"))
            snippet = self._select_relevant_snippet(chunk.text, query_terms)
            overlap_count = len(query_terms.intersection(self._normalize_terms(chunk.text)))

            citations.append(
                SourceCitation(
                    source_name=source_name,
                    page_number=page_number,
                    retrieval_method=result.retrieval_method,
                    score=float(result.score),
                    overlap_count=overlap_count,
                    snippet=snippet,
                )
            )

        if not citations:
            return ExplainabilityReport(summary="No retrieval evidence is available.", resolved_query=(resolved_query or ""), summary_used=(summary_used or ""), memory_hits=memory_hits)

        summary = self._build_summary(citations)
        query_alignment = self._build_query_alignment(query, citations)

        return ExplainabilityReport(citations=citations, summary=summary, query_alignment=query_alignment, resolved_query=(resolved_query or ""), summary_used=(summary_used or ""), memory_hits=memory_hits)

    def _normalize_terms(self, text: str) -> set[str]:
        return {token for token in re.findall(r"\w+", text.lower()) if token}

    def _select_relevant_snippet(self, text: str, query_terms: set[str]) -> str:
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        best_sentence = ""
        best_overlap = -1

        for sentence in sentences:
            overlap = len(query_terms.intersection(self._normalize_terms(sentence)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence

        if not best_sentence:
            best_sentence = sentences[0] if sentences else text

        snippet = best_sentence.strip()
        return snippet if len(snippet) <= 220 else snippet[:217].rstrip() + "..."

    def _build_summary(self, citations: List[SourceCitation]) -> str:
        top_sources = sorted(citations, key=lambda citation: (-citation.score, -citation.overlap_count))[:3]
        source_names = [f"{citation.source_name} ({citation.retrieval_method})" for citation in top_sources]
        return (
            "The answer is grounded in retrieved passages from the top sources listed below. "
            "Each source is ranked by retrieval relevance and alignment with your query."
        )

    def _build_query_alignment(self, query: str, citations: List[SourceCitation]) -> str:
        if not query:
            return ""

        relevant_sources = [citation for citation in citations if citation.overlap_count > 0]
        if not relevant_sources:
            return "The retrieved sources contain limited direct query overlap, so the answer relies more on general context."

        return (
            f"{len(relevant_sources)} of the retrieved sources contain direct query term overlap, "
            f"indicating the response is supported by relevant passages."
        )
