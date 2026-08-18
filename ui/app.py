"""Main Streamlit application for IITB Insti-Assist Pro."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any
from uuid import uuid4

import streamlit as st

from config import (
    EMBEDDING_MODEL_NAME,
    LLM_PROVIDER,
    LLM_MODEL,
    PROJECT_NAME,
    TOP_K,
    VECTORSTORE_INDEX_FILE,
    VECTORSTORE_METADATA_FILE,
)
from evaluation.response_validator import ResponseValidator
from generation.generator import Generator
from llm.factory import LLMFactory
from rag.explainability import ExplainabilityBuilder
from retrieval.retrieval_pipeline import RetrievalPipeline
from retrieval.retriever import Retriever
from ui.components import render_assistant_response, render_message_bubble
from ui.sidebar import render_sidebar
from ui.styles import inject_styles
from vectorstore.faiss_store import FAISSStore
from ui.document_manager_ui import render_document_manager_sidebar
from memory.conversation_memory import ConversationEntry
from memory.memory_manager import get_memory_manager
from observability import TraceContext, get_current_trace
from ui.memory_ui import render_memory_sidebar
from utils.logging_utils import setup_logging

logger = setup_logging("app.log")


@st.cache_resource
def _load_vector_store() -> tuple[FAISSStore, bool, str]:
    """Load the FAISS index and metadata from disk for startup diagnostics."""
    store = FAISSStore(index_path=VECTORSTORE_INDEX_FILE, metadata_path=VECTORSTORE_METADATA_FILE)
    try:
        store.load()
        return store, True, "Vector store loaded successfully."
    except FileNotFoundError as exc:
        return store, False, str(exc)
    except Exception as exc:
        return store, False, f"Vector store load failed: {exc}"


@st.cache_resource
def _build_pipeline() -> tuple[RetrievalPipeline, Generator]:
    """Create reusable backend pipeline objects for the UI."""
    vector_store, index_available, _ = _load_vector_store()
    retriever = Retriever(vector_store=vector_store) if index_available else None
    retrieval_pipeline = RetrievalPipeline(retriever=retriever)
    generator = Generator()
    return retrieval_pipeline, generator


def _initialize_session() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "startup_info" not in st.session_state:
        st.session_state.startup_info = {}
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = 0
    if "index_chunks" not in st.session_state:
        st.session_state.index_chunks = 0
    if "index_available" not in st.session_state:
        st.session_state.index_available = False
    if "llm_available" not in st.session_state:
        st.session_state.llm_available = False
    if "session_id" not in st.session_state:
        # The memory manager is shared by the Streamlit process, so each browser
        # session needs its own key to keep conversation context isolated.
        st.session_state.session_id = str(uuid4())


def _append_interaction(user_message: str, generated_answer: object, retrieval_results: list[object]) -> None:
    st.session_state.chat_history.append(user_message)
    st.session_state.responses.append(
        {
            "answer": generated_answer,
            "retrieval_results": retrieval_results,
        }
    )


def _build_startup_info() -> dict[str, Any]:
    """Collect startup readiness signals for index and LLM configuration."""
    vector_store, index_available, index_message = _load_vector_store()
    try:
        llm = LLMFactory.create()
        llm_available = llm.health_check()
        provider_name = llm.provider_name()
        model_name = llm.model_name
        provider_message = "LLM provider is configured." if llm_available else f"{llm.api_key_env} is not configured."
    except ValueError as exc:
        llm_available = False
        provider_name = LLM_PROVIDER
        model_name = LLM_MODEL
        provider_message = str(exc)
        logger.error("LLM provider configuration error: %s", exc)
    vector_count = int(vector_store.index.ntotal) if index_available and vector_store.index is not None else 0

    return {
        "llm_available": llm_available,
        "llm_provider": provider_name,
        "llm_model": model_name,
        "llm_message": provider_message,
        "index_available": index_available,
        "index_message": index_message,
        "metadata_count": len(getattr(vector_store, "metadata", [])) if index_available else 0,
        "vector_count": vector_count,
        "vector_store_path": str(VECTORSTORE_INDEX_FILE),
    }


def _update_startup_state(startup_info: dict[str, Any]) -> None:
    st.session_state.startup_info = startup_info
    st.session_state.indexed_documents = startup_info.get("metadata_count", 0)
    st.session_state.index_chunks = startup_info.get("vector_count", 0)
    st.session_state.index_available = startup_info.get("index_available", False)
    st.session_state.llm_available = startup_info.get("llm_available", False)
    st.session_state.llm_provider = startup_info.get("llm_provider", LLM_PROVIDER)


def _render_startup_panel(startup_info: dict[str, Any]) -> None:
    with st.expander("Startup diagnostics", expanded=True):
        if startup_info["index_available"]:
            st.success("FAISS index is available and ready for retrieval.")
            st.markdown(f"**Vector store path:** `{startup_info['vector_store_path']}`")
            st.markdown(f"**Indexed chunks:** {startup_info['vector_count']}")
        else:
            st.error("FAISS vector store is not available.")
            st.markdown(startup_info["index_message"])

        if startup_info["llm_available"]:
            st.success(
                f"LLM provider: {startup_info['llm_provider']} ({startup_info['llm_model']})."
            )
        else:
            st.warning(startup_info["llm_message"])


def _build_memory_context(manager, session_id: str, original_query: str) -> tuple[str, dict[str, Any]]:
    """Resolve the query and build the active memory context."""
    memory_start = time.perf_counter()
    recent_entries = manager.get_memory(session_id).recent(8)
    followup_detected = manager.detector.is_followup(original_query, recent_entries)
    resolved_query = manager.resolve_query(session_id, original_query)
    context = manager.build_context(session_id)
    retrieval_elapsed_ms = (time.perf_counter() - memory_start) * 1000
    pronoun_resolved = resolved_query.strip().casefold() != original_query.strip().casefold()
    context["resolved_query"] = resolved_query
    context["original_query"] = original_query
    context["memory_retrieval_time_ms"] = retrieval_elapsed_ms
    context["followup_detected"] = followup_detected
    context["pronoun_resolved"] = pronoun_resolved
    logger.info(
        "Memory context built for session %s | original_query=%s | resolved_query=%s | followup_detected=%s | pronoun_resolved=%s | summary_length=%s | recent_context_size=%s | memory_retrieval_time_ms=%.2f",
        session_id,
        original_query,
        resolved_query,
        context["followup_detected"],
        context["pronoun_resolved"],
        context.get("summary_length", 0),
        context.get("recent_context_size", 0),
        retrieval_elapsed_ms,
    )
    logger.info(
        "Conversation summary updated for session %s | summary_length=%s | recent_messages=%s",
        session_id,
        context.get("summary_length", 0),
        len(context.get("recent_messages", [])),
    )
    return resolved_query, context


def _store_memory_turn(
    manager,
    session_id: str,
    original_query: str,
    resolved_query: str,
    answer: Any,
    retrieval_results: list[Any],
    context: dict[str, Any],
) -> None:
    retrieved_chunks = []
    retrieved_doc_ids = []
    for result in retrieval_results:
        chunk = result.chunk
        metadata = getattr(chunk, "metadata", {}) or {}
        document_id = metadata.get("document_id") or getattr(chunk, "document_id", None)
        if document_id:
            retrieved_doc_ids.append(str(document_id))
        retrieved_chunks.append(
            {
                "chunk_id": getattr(chunk, "chunk_id", ""),
                "document_id": document_id,
                "source": getattr(chunk, "source", ""),
                "text": getattr(chunk, "text", ""),
            }
        )

    manager.get_memory(session_id).add_entry(
        ConversationEntry(
            user_query=original_query,
            assistant_response=answer.answer,
            retrieved_chunks=retrieved_chunks,
            retrieved_document_ids=sorted(set(retrieved_doc_ids)),
            sources=list(answer.sources),
            confidence=answer.confidence_score,
            original_query=original_query,
            resolved_query=resolved_query,
            citations=[
                citation.source_name
                for citation in getattr(answer.explainability_report, "citations", [])
            ]
            or list(answer.sources),
            summary_used=context.get("summary", ""),
            followup_detected=context.get("followup_detected", False),
            pronoun_resolved=context.get("pronoun_resolved", False),
            memory_summary_length=int(context.get("summary_length", 0)),
            recent_context_size=int(context.get("recent_context_size", 0)),
        )
    )
    logger.info(
        "Conversation memory updated for session %s | original_query=%s | resolved_query=%s | chunks=%s | document_ids=%s | citations=%s | confidence=%.3f",
        session_id,
        original_query,
        resolved_query,
        len(retrieved_chunks),
        len(set(retrieved_doc_ids)),
        len(getattr(answer.explainability_report, "citations", []) or answer.sources),
        float(answer.confidence_score),
    )


def _process_conversational_turn(
    manager,
    session_id: str,
    original_query: str,
    retrieval_pipeline,
    generator,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Resolve query, retrieve context, generate an answer, and enrich memory diagnostics."""
    trace_scope = nullcontext() if get_current_trace() is not None else TraceContext(session_id=session_id)
    with trace_scope:
        resolved_query, retrieval_results, context = _prepare_conversational_turn(
            manager,
            session_id,
            original_query,
            retrieval_pipeline,
        )
        generated_answer = generator.generate(resolved_query, retrieval_results, context=context)
        _finalize_conversational_turn(
            manager,
            session_id,
            original_query,
            resolved_query,
            generated_answer,
            retrieval_results,
            context,
        )
    return generated_answer, retrieval_results, context


def _prepare_conversational_turn(
    manager,
    session_id: str,
    original_query: str,
    retrieval_pipeline,
) -> tuple[str, list[Any], dict[str, Any]]:
    """Resolve and retrieve once for either synchronous or streaming generation."""
    resolved_query, context = _build_memory_context(manager, session_id, original_query)
    retrieval_results = retrieval_pipeline.retrieve(resolved_query, top_k=5)
    return resolved_query, retrieval_results, context


def _finalize_conversational_turn(
    manager,
    session_id: str,
    original_query: str,
    resolved_query: str,
    generated_answer: Any,
    retrieval_results: list[Any],
    context: dict[str, Any],
) -> None:
    """Evaluate and persist a completed answer; never call this for partial streaming text."""
    evaluation_result = ResponseValidator().validate(generated_answer, retrieval_results)
    generated_answer.confidence_score = evaluation_result.confidence_score
    generated_answer.unsupported_claims = evaluation_result.unsupported_claims
    generated_answer.original_query = original_query
    generated_answer.resolved_query = resolved_query
    generated_answer.summary_used = context.get("summary")
    generated_answer.recent_messages_count = len(context.get("recent_messages", []))
    generated_answer.memory_hits = len(context.get("recent_messages", []))
    generated_answer.followup_detected = bool(context.get("followup_detected", False))
    generated_answer.pronoun_resolved = bool(context.get("pronoun_resolved", False))
    generated_answer.memory_summary_length = int(context.get("summary_length", 0))
    generated_answer.recent_context_size = int(context.get("recent_context_size", 0))
    generated_answer.memory_retrieval_time_ms = float(context.get("memory_retrieval_time_ms", 0.0))

    explain_builder = ExplainabilityBuilder()
    generated_answer.explainability_report = explain_builder.build(
        original_query,
        retrieval_results,
        resolved_query=resolved_query,
        summary_used=context.get("summary"),
        memory_hits=generated_answer.memory_hits,
    )
    _store_memory_turn(
        manager,
        session_id,
        original_query,
        resolved_query,
        generated_answer,
        retrieval_results,
        context,
    )


def _render_streaming_answer(answer_stream) -> Any:
    """Render a temporary typing effect and return its completed generated answer."""
    stream_placeholder = st.empty()
    with stream_placeholder.container():
        st.write_stream(answer_stream)
    stream_placeholder.empty()
    if answer_stream.final_answer is None:
        raise RuntimeError("Streaming response ended before a complete answer was available.")
    return answer_stream.final_answer


def main() -> None:
    """Render the Streamlit application."""
    inject_styles()
    st.set_page_config(page_title=PROJECT_NAME, page_icon="🎓", layout="wide")

    _initialize_session()
    startup_info = _build_startup_info()
    _update_startup_state(startup_info)
    render_sidebar()
    render_memory_sidebar()
    render_document_manager_sidebar()

    st.markdown("<div class='page-header'><h1>IITB Insti-Assist Pro</h1></div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='lead'>Ask questions about IIT Bombay documents and institutional information.</p>",
        unsafe_allow_html=True,
    )

    _render_startup_panel(startup_info)

    if not startup_info["index_available"]:
        st.warning(
            "The application is not fully ready because the FAISS vector store is missing."
        )
        st.info(
            "Create or load a valid FAISS index into `data/vectorstore/` and then reload the app."
        )
        return

    chat_area = st.container()
    with chat_area:
        for index, user_message in enumerate(st.session_state.chat_history):
            render_message_bubble("user", user_message)
            if index < len(st.session_state.responses):
                response = st.session_state.responses[index]
                render_assistant_response(response["answer"], response["retrieval_results"])

    prompt_text = st.chat_input("Ask about IIT Bombay documents...")
    if prompt_text:
        st.session_state.chat_history.append(prompt_text)
        render_message_bubble("user", prompt_text)

        manager = get_memory_manager(storage_dir="data/memory")
        session_id = st.session_state.session_id
        original_query = prompt_text

        retrieval_pipeline, generator = _build_pipeline()
        try:
            with TraceContext(session_id=session_id) as trace:
                with st.spinner("Retrieving response context..."):
                    resolved_query, retrieval_results, context = _prepare_conversational_turn(
                        manager,
                        session_id,
                        original_query,
                        retrieval_pipeline,
                    )

                answer_stream = generator.stream(resolved_query, retrieval_results, context=context)
                generated_answer = _render_streaming_answer(answer_stream)
                _finalize_conversational_turn(
                    manager,
                    session_id,
                    original_query,
                    resolved_query,
                    generated_answer,
                    retrieval_results,
                    context,
                )
                st.session_state.last_trace_id = trace.trace_id

            _append_interaction(prompt_text, generated_answer, retrieval_results)
            render_assistant_response(generated_answer, retrieval_results)
        except Exception as exc:
            st.warning("Sorry, we could not generate a response right now.")
            st.error(str(exc))


if __name__ == "__main__":
    main()
