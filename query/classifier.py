"""Rule-based query classification for routing and context awareness."""

from __future__ import annotations

import re
from typing import Final

from utils.logging_utils import setup_logging

logger = setup_logging("query.log")


class QueryClassifier:
    """Classify queries into predefined categories using rule-based patterns."""

    _CATEGORY_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
        "Clubs": re.compile(r"\b(club|society|WnCC|SAC|coding|arts|drama|sports)\b", re.IGNORECASE),
        "Hostel": re.compile(r"\b(hostel|mess|room|accommodation|dorm|fees?)\b", re.IGNORECASE),
        "Placement": re.compile(r"\b(placement|internship|career|company|recruitment|job|salary)\b", re.IGNORECASE),
        "Campus": re.compile(r"\b(campus|location|transport|facilities|gym|library|canteen)\b", re.IGNORECASE),
        "Academic": re.compile(r"\b(registration|course|semester|academic|exam|grades|syllabus)\b", re.IGNORECASE),
        "Administration": re.compile(r"\b(administration|office|authority|policy|document|fee|tuition)\b", re.IGNORECASE),
        "General": re.compile(r"\b(help|support|information|faq|general)\b", re.IGNORECASE),
    }

    def classify(self, query: str) -> str:
        """Return a best-effort category label for a query."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string for classification.")

        cleaned_query = query.strip()
        for category, pattern in self._CATEGORY_PATTERNS.items():
            if pattern.search(cleaned_query):
                logger.info("Classified query '%s' as %s", query, category)
                return category

        logger.info("Classified query '%s' as General", query)
        return "General"
