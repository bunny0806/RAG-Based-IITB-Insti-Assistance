"""Streamlit UI components for the Document Manager sidebar and pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import VECTORSTORE_DIR
from documents.document_manager import DocumentManager
from utils.logging_utils import setup_logging

logger = setup_logging("ui_documents.log")


@st.cache_resource
def _build_doc_manager() -> DocumentManager:
    raw_dir = Path("data") / "raw"
    index_path = Path(VECTORSTORE_DIR) / "faiss.index"
    metadata_path = Path(VECTORSTORE_DIR) / "faiss.json"

    manager = DocumentManager(
        raw_dir=raw_dir,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    return manager


def render_document_manager_sidebar() -> None:
    st.markdown(
        "<div class='sidebar-section-title'>Document Manager</div>",
        unsafe_allow_html=True,
    )

    manager = _build_doc_manager()

    # Upload Section
    with st.expander("Upload PDF"):

        uploaded = st.file_uploader(
            "Choose a PDF to upload",
            type=["pdf"],
            key="doc_upload",
        )

        if uploaded is not None:
            result = manager.upload(uploaded)

            metadata = result.get("metadata")
            created = result.get("created")

            if created:
                st.success(
                    f"Uploaded {metadata.filename} as {metadata.document_id}"
                )
            else:
                st.info(
                    f"Duplicate detected, file already uploaded as "
                    f"{metadata.document_id}"
                )

    # Refresh metadata
    if st.button("Refresh metadata"):

        # Streamlit >= 1.30 replacement for experimental_rerun()
        st.rerun()


    # Uploaded documents
    st.markdown(
        "<div class='section-header'>Uploaded Documents</div>",
        unsafe_allow_html=True,
    )

    docs = manager.list_documents()

    for doc in docs:

        cols = st.columns([3, 1, 1, 1])

        with cols[0]:
            st.markdown(
                f"""
                **{doc.title}**
                <br>
                <small>{doc.filename}</small>
                """,
                unsafe_allow_html=True,
            )

        with cols[1]:
            st.markdown(
                f"<div class='metric-card'>{doc.status}</div>",
                unsafe_allow_html=True,
            )

        with cols[2]:

            if st.button(
                "Reindex",
                key=f"reindex_{doc.document_id}",
            ):
                try:
                    manager.reindex(doc.document_id)
                    st.success("Reindex completed")
                except Exception as exc:
                    st.error(str(exc))


        with cols[3]:

            if st.button(
                "Delete",
                key=f"delete_{doc.document_id}",
            ):
                try:
                    manager.delete(doc.document_id)
                    st.success("Deleted document")
                    st.rerun()

                except Exception as exc:
                    st.error(str(exc))


    # Reindex all
    if st.button("Reindex All"):

        try:
            manager.reindex()
            st.success("Reindex all completed")
            st.rerun()

        except Exception as exc:
            st.error(str(exc))


    # Statistics
    stats = manager.stats()

    st.markdown(
        "<div class='section-header'>Index Statistics</div>",
        unsafe_allow_html=True,
    )

    st.metric(
        "Total Documents",
        stats.get("total_documents", 0),
    )

    st.metric(
        "Total Chunks",
        stats.get("total_chunks", 0),
    )

    st.metric(
        "Avg Chunks/Document",
        f"{stats.get('average_chunks_per_document', 0):.1f}",
    )

    st.metric(
        "Index Size (vectors)",
        stats.get("index_size", 0),
    )

    st.metric(
        "Vector Dim",
        stats.get("vector_dimension", 0),
    )

    st.metric(
        "Embedding Model",
        stats.get("embedding_model", "n/a"),
    )