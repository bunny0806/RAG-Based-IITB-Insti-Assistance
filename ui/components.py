"""Reusable UI components for the Streamlit chat interface."""

from __future__ import annotations

from typing import Sequence

import streamlit as st

from generation.generator import GeneratedAnswer
from rag.explainability import ExplainabilityReport
from retrieval.models import RetrievalResult


def render_message_bubble(role: str, content: str) -> None:
    """Render a chat message bubble in the Streamlit chat UI."""
    with st.chat_message(role):
        st.markdown(content)


def render_assistant_response(answer: GeneratedAnswer, retrieval_results: Sequence[RetrievalResult]) -> None:
    """Render a structured assistant response with answer, sources, and metrics."""
    confidence = _derive_confidence(answer)
    badge_color = _badge_color(confidence)
    badge_label = _badge_label(confidence)

    with st.container():
        st.markdown(
            f"<div class='assistant-card'><div class='assistant-header'><span class='assistant-badge' style='background:{badge_color};'>{badge_label}</span></div><div class='assistant-content'>{answer.answer}</div></div>",
            unsafe_allow_html=True,
        )

        if answer.explainability_report is not None:
            st.markdown("<div class='section-header'>Explainability</div>", unsafe_allow_html=True)
            _render_explainability_summary(answer.explainability_report)
            _render_explainability_citations(answer.explainability_report)

            # Conversation context section
            with st.expander("Conversation Context"):
                st.write(f"Original Query: {getattr(answer, 'original_query', 'N/A')}")
                st.write(f"Resolved Query: {getattr(answer, 'resolved_query', 'N/A')}")
                st.write(f"Summary Used: {getattr(answer, 'summary_used', 'N/A')}")
                st.write(f"Recent Messages Count: {getattr(answer, 'recent_messages_count', 0)}")
                st.write(f"Follow-up Detected: {getattr(answer, 'followup_detected', False)}")
                st.write(f"Pronoun Resolved: {getattr(answer, 'pronoun_resolved', False)}")
                st.write(f"Memory Summary Length: {getattr(answer, 'memory_summary_length', 0)}")
                st.write(f"Recent Context Size: {getattr(answer, 'recent_context_size', 0)}")
                st.write(f"Memory Retrieval Time (ms): {getattr(answer, 'memory_retrieval_time_ms', 0.0):.2f}")

        if retrieval_results:
            st.markdown("<div class='section-header'>Sources</div>", unsafe_allow_html=True)
            for result in retrieval_results:
                _render_source_card(result)
        else:
            st.warning("No retrieved sources are available for this response.")

        st.markdown("<div class='section-header'>Response Metrics</div>", unsafe_allow_html=True)
        metrics = {
            "Latency": f"{answer.latency:.1f} ms",
            "Retrieved Chunks": len(retrieval_results),
            "Grounded": "Yes" if answer.grounded else "No",
            "Embedding Model": "all-MiniLM-L6-v2",
        }
        cols = st.columns(4)
        for col, (label, value) in zip(cols, metrics.items()):
            with col:
                st.markdown(
                    f"<div class='metric-card'><strong>{label}</strong><div class='metric-value'>{value}</div></div>",
                    unsafe_allow_html=True,
                )


def _render_explainability_summary(report: ExplainabilityReport) -> None:
    st.markdown(
        f"<div class='explainability-card'><strong>Summary:</strong> {report.summary}<br><strong>Query alignment:</strong> {report.query_alignment}</div>",
        unsafe_allow_html=True,
    )


def _render_explainability_citations(report: ExplainabilityReport) -> None:
    if not report.citations:
        st.info("No citations were identified for this response.")
        return

    for citation in report.citations[:3]:
        st.markdown(
            "<div class='source-card'>"
            f"<div class='source-header'>📌 <strong>{citation.source_name}</strong></div>"
            f"<div class='source-meta'>{citation.retrieval_method} · Score {citation.score:.2f} · Overlap {citation.overlap_count}</div>"
            f"<div class='source-snippet'>{citation.snippet}</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_source_card(result: RetrievalResult) -> None:
    """Render a source card for a retrieved chunk."""
    chunk = result.chunk
    metadata = getattr(chunk, "metadata", {}) or {}
    source_name = metadata.get("document_name") or metadata.get("filename") or metadata.get("title") or chunk.source or "Unknown document"
    page_number = metadata.get("page_number", "n/a")
    similarity = f"{result.score:.2f}"

    with st.container():
        st.markdown(
            "<div class='source-card'>"
            f"<div class='source-header'>📄 <strong>{source_name}</strong></div>"
            f"<div class='source-meta'>Page {page_number} · Similarity {similarity}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.expander("View retrieved chunk"):
            st.write(chunk.text)


def _derive_confidence(answer: GeneratedAnswer) -> float:
    """Derive a proxy confidence score for UI display."""
    if getattr(answer, "confidence_score", None) is not None:
        return float(answer.confidence_score)
    return 0.85 if answer.grounded else 0.35


def _badge_label(confidence: float) -> str:
    if confidence >= 0.8:
        return "High confidence"
    if confidence >= 0.5:
        return "Medium confidence"
    return "Low confidence"


def _badge_color(confidence: float) -> str:
    if confidence >= 0.8:
        return "#16a34a"
    if confidence >= 0.5:
        return "#f59e0b"
    return "#dc2626"
