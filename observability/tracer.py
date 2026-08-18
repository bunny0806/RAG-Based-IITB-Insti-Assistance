"""Request-scoped tracing with stage timing and structured lifecycle events."""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional
from uuid import uuid4

from observability.logger import log_event
from observability.metrics import get_metrics_collector

TRACE_STAGES = (
    "query_processing",
    "memory_resolution",
    "retrieval",
    "reranking",
    "prompt_building",
    "generation",
    "evaluation",
    "storage",
)


@dataclass
class StageRecord:
    stage: str
    start_time: float
    end_time: float | None = None
    duration: float = 0.0
    status: str = "running"
    error_message: str | None = None


@dataclass
class TraceContext:
    """Trace one user request and its ordered RAG lifecycle stages."""

    session_id: str
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stages: list[StageRecord] = field(default_factory=list)
    _start_time: float = field(default_factory=time.perf_counter, init=False, repr=False)
    _token: Token | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "TraceContext":
        self._start_time = time.perf_counter()
        self._token = _CURRENT_TRACE.set(self)
        log_event("trace_started", trace_id=self.trace_id, session_id=self.session_id)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        total_latency = (time.perf_counter() - self._start_time) * 1000
        status = "failed" if exc is not None else "success"
        get_metrics_collector().complete(self.trace_id, total_latency, status)
        log_event(
            "trace_completed",
            trace_id=self.trace_id,
            session_id=self.session_id,
            total_latency=total_latency,
            status=status,
            error_message=str(exc) if exc is not None else None,
        )
        if self._token is not None:
            _CURRENT_TRACE.reset(self._token)
        return False

    @contextmanager
    def stage(self, stage_name: str) -> Iterator[StageRecord]:
        record = StageRecord(stage=stage_name, start_time=time.perf_counter())
        self.stages.append(record)
        try:
            yield record
        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)
            raise
        else:
            record.status = "success"
        finally:
            record.end_time = time.perf_counter()
            record.duration = (record.end_time - record.start_time) * 1000
            get_metrics_collector().record_stage(self.trace_id, stage_name, record.duration, record.status)
            log_event(
                "stage_completed",
                trace_id=self.trace_id,
                session_id=self.session_id,
                stage=stage_name,
                start_time=record.start_time,
                end_time=record.end_time,
                latency=record.duration,
                status=record.status,
                error_message=record.error_message,
            )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "stages": [asdict(stage) for stage in self.stages],
        }


_CURRENT_TRACE: ContextVar[Optional[TraceContext]] = ContextVar("current_rag_trace", default=None)


def get_current_trace() -> TraceContext | None:
    return _CURRENT_TRACE.get()


@contextmanager
def trace_stage(stage_name: str) -> Iterator[StageRecord | None]:
    """Time a stage only when a request trace is active."""
    trace = get_current_trace()
    if trace is None:
        yield None
        return
    with trace.stage(stage_name) as record:
        yield record
