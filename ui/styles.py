"""Shared styling helpers for the Streamlit interface."""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Inject a polished, professional theme for the app."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f9ff;
            --panel: #ffffff;
            --accent: #2563eb;
            --accent-soft: #dbeafe;
            --text: #0f172a;
            --muted: #64748b;
            --success: #16a34a;
            --warning: #f59e0b;
            --danger: #dc2626;
        }
        body {
            background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
            border-right: 1px solid #dbeafe;
        }
        .sidebar-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .sidebar-subtitle,
        .sidebar-text {
            color: #475569;
            margin-bottom: 1rem;
        }
        .sidebar-section-title {
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.5rem;
        }
        .assistant-card {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 16px 40px rgba(37, 99, 235, 0.08);
        }
        .assistant-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .assistant-badge {
            color: #ffffff;
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .assistant-content {
            color: #0f172a;
            line-height: 1.75;
            white-space: pre-wrap;
        }
        .source-card {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .source-header {
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 6px;
        }
        .source-meta {
            color: #64748b;
            margin-bottom: 12px;
            font-size: 0.95rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 12px;
            text-align: left;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.05rem;
            margin-top: 6px;
            display: block;
        }
        .section-header {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 20px;
            margin-bottom: 12px;
        }
        .lead {
            color: #475569;
            margin-bottom: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
