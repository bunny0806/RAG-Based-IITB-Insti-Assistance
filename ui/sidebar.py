"""Sidebar layout and controls for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from config import EMBEDDING_MODEL_NAME, LLM_MODEL, LLM_PROVIDER, PROJECT_NAME, TOP_K
from observability import get_metrics_collector


def render_sidebar() -> None:
    """Render the app sidebar with metadata and controls."""
    with st.sidebar:
        st.markdown(f"<div class='sidebar-title'>{PROJECT_NAME}</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-subtitle'>IIT Bombay assistant powered by retrieval-augmented generation.</div>", unsafe_allow_html=True)

        st.markdown("<hr />", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section-title'>Project Overview</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-text'>A clean, professional interface for retrieving IITB institutional documents and policies.</div>", unsafe_allow_html=True)

        st.markdown("<hr />", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section-title'>Statistics</div>", unsafe_allow_html=True)
        st.metric("Index Status", "Ready" if st.session_state.get("index_available", False) else "Unavailable")
        st.metric("Indexed Documents", st.session_state.get("indexed_documents", 0))
        st.metric("Chunks", st.session_state.get("index_chunks", 0))

        st.markdown("<hr />", unsafe_allow_html=True)
        st.metric("Embedding Model", EMBEDDING_MODEL_NAME)
        st.metric("LLM Provider", st.session_state.get("llm_provider", LLM_PROVIDER))
        st.metric("LLM Model", LLM_MODEL)
        st.metric("Top K", TOP_K)
        st.metric("LLM API", "Configured" if st.session_state.get("llm_available", False) else "Missing")

        st.markdown("<hr />", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section-title'>System Health</div>", unsafe_allow_html=True)
        health = get_metrics_collector().summary()
        st.metric("Current Provider", health.get("provider_name") or st.session_state.get("llm_provider", LLM_PROVIDER))
        st.metric("Average Latency", f"{health['average_latency']:.1f} ms")
        st.metric("Last Retrieval Score", f"{health['last_retrieval_score']:.3f}")
        st.metric("Last Confidence Score", f"{health['last_confidence_score']:.3f}")
        st.metric("Requests Processed", health["requests_processed"])

        st.markdown("<hr />", unsafe_allow_html=True)
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.responses = []
            st.success("Chat history cleared.")
