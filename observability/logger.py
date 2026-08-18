"""Structured JSON logging for RAG observability events."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from config import LOGS_DIR


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("iitb_insti_assist.observability")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOGS_DIR / "observability.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(event: str, **fields: object) -> None:
    """Write one machine-readable event per line without logging prompt contents.

    Monitoring must be non-intrusive: a transient log-directory or file error
    must not fail the RAG request that is being observed.
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    try:
        _get_logger().info(json.dumps(payload, default=str, ensure_ascii=False))
    except (OSError, ValueError):
        logging.getLogger(__name__).debug("Unable to persist observability event", exc_info=True)
