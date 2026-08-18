"""Streamlit UI components for the conversational memory sidebar."""

from __future__ import annotations

import streamlit as st
from uuid import uuid4

from memory.memory_manager import get_memory_manager


def render_memory_sidebar() -> None:
    manager = get_memory_manager(storage_dir="data/memory")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())

    session_id = st.session_state.session_id

    with st.sidebar.expander("Conversation Memory", expanded=True):
        st.write("Session:", session_id)
        mem = manager.get_memory(session_id)
        context = manager.build_context(session_id)
        st.metric("Conversation length", mem.length())
        st.metric("Summary size (chars)", context.get("summary_length", 0))
        st.metric("Active context size (chars)", context.get("active_context_size", 0))
        st.markdown("**Summary preview**")
        st.write(context.get("summary", "(empty)"))

        if st.button("Clear Conversation"):
            manager.clear_memory(session_id)
            st.experimental_rerun()
