"""Utilities for preprocessing and normalizing user queries."""

from __future__ import annotations

import re
from typing import Final

from utils.logging_utils import setup_logging

logger = setup_logging("retrieval.log")


class QueryProcessor:
    """Normalize and lightly expand user queries without changing their intent."""

    _WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
    _ABBREVIATION_MAP: Final[dict[str, str]] = {
        "wncc": "Web and Coding Club (WnCC)",
        "iitb": "IIT Bombay",
        "iit": "Indian Institute of Technology",
        "hostel": "hostel",
    }

    def process(self, query: str) -> str:
        """Clean and normalize a user query."""
        if not isinstance(query, str):
            raise TypeError("QueryProcessor expects a string query.")

        cleaned_query = query.strip()
        cleaned_query = self._WHITESPACE_RE.sub(" ", cleaned_query)
        cleaned_query = self._expand_abbreviations(cleaned_query)

        logger.info("Query preprocessing complete: %s", cleaned_query)
        return cleaned_query

    def _expand_abbreviations(self, text: str) -> str:
        """Expand simple abbreviations where useful and safe."""
        expanded_text = text
        for abbreviation, expansion in self._ABBREVIATION_MAP.items():
            pattern = re.compile(rf"\b{re.escape(abbreviation)}\b", re.IGNORECASE)
            expanded_text = pattern.sub(expansion, expanded_text)
        return expanded_text
