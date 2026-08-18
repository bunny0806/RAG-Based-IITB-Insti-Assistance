"""Text cleaning utilities for preprocessing documents."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from utils.logging_utils import setup_logging

logger = setup_logging("preprocessing.log")


class TextCleaner:
    """Clean text conservatively without removing semantic content."""

    _MULTI_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
    _LEADING_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"^\s+", re.MULTILINE)
    _TRAILING_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+$", re.MULTILINE)

    def clean(self, text: str) -> str:
        """Normalize text while preserving meaningful structure and punctuation."""
        if not isinstance(text, str):
            raise TypeError("TextCleaner expects a string input.")

        normalized_text = unicodedata.normalize("NFKC", text)
        normalized_text = self._MULTI_SPACE_RE.sub(" ", normalized_text)
        normalized_text = self._MULTI_NEWLINE_RE.sub("\n\n", normalized_text)
        normalized_text = self._LEADING_SPACE_RE.sub("", normalized_text)
        normalized_text = self._TRAILING_SPACE_RE.sub("", normalized_text)

        cleaned_text = normalized_text.strip()
        if not cleaned_text:
            logger.warning("Received empty text during cleaning.")
            return ""

        logger.info("Document cleaned successfully.")
        return cleaned_text
