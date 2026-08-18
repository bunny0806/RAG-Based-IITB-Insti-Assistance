"""Query rewriting for ambiguous or terse user inputs."""

from __future__ import annotations

import re
from typing import Final

from utils.logging_utils import setup_logging

logger = setup_logging("query.log")


class QueryRewriter:
    """Rewrite queries using rule-based templates while preserving user intent."""

    _REWRITE_MAP: Final[dict[str, str]] = {
        r"\bregistration\b": "IIT Bombay academic course registration",
        r"\bhostel fees\b": "IIT Bombay hostel fee payment process",
        r"\bwncc\b": "Web and Coding Club IIT Bombay",
        r"\bsac\b": "Student Activity Centre IIT Bombay",
        r"\bug\b": "Undergraduate programs IIT Bombay",
        r"\bpg\b": "Postgraduate programs IIT Bombay",
    }

    def rewrite(self, query: str) -> str:
        """Rewrite a query if it matches a known ambiguous pattern."""
        if not isinstance(query, str):
            raise TypeError("Query must be a string for rewriting.")

        rewritten_query = query.strip()
        for pattern, rewrite_text in self._REWRITE_MAP.items():
            if re.search(pattern, rewritten_query, re.IGNORECASE):
                logger.info("Rewrote query '%s' to '%s'", query, rewrite_text)
                return rewrite_text

        logger.info("No rewrite rule applied to query '%s'", query)
        return rewritten_query
