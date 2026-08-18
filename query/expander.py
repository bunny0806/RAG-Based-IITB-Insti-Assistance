"""Abbreviation expansion for user search queries."""

from __future__ import annotations

import re
from typing import Final

from utils.logging_utils import setup_logging

logger = setup_logging("query.log")


class QueryExpander:
    """Expand abbreviations using a configurable dictionary."""

    _ABBREVIATION_MAP: Final[dict[str, str]] = {
        "iitb": "IIT Bombay",
        "iit": "Indian Institute of Technology",
        "sac": "Student Activity Centre",
        "ug": "Undergraduate",
        "pg": "Postgraduate",
        "wncc": "Web and Coding Club",
    }

    def expand(self, query: str) -> str:
        """Expand abbreviations in the original query."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string for expansion.")

        expanded_query = query.strip()
        sorted_abbreviations = sorted(self._ABBREVIATION_MAP.keys(), key=len, reverse=True)
        pattern = re.compile(r"\b(" + "|".join(re.escape(abbrev) for abbrev in sorted_abbreviations) + r")\b", re.IGNORECASE)

        def replace(match: re.Match[str]) -> str:
            return self._ABBREVIATION_MAP[match.group(1).lower()]

        expanded_query = pattern.sub(replace, expanded_query)
        expanded_query = re.sub(r"\s+", " ", expanded_query).strip()
        logger.info("Expanded query '%s' to '%s'", query, expanded_query)
        return expanded_query
