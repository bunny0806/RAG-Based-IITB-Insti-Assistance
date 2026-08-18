"""Thread-safe request metrics collection and UI-friendly summaries."""

from __future__ import annotations

from threading import Lock
from typing import Any


class MetricsCollector:
    """Collect trace-scoped metrics while retaining compact aggregate health data."""

    _STAGE_METRICS = {
        "retrieval": "retrieval_latency",
        "generation": "generation_latency",
        "evaluation": "evaluation_latency",
    }

    def __init__(self) -> None:
        self._requests: dict[str, dict[str, Any]] = {}
        self._completed: list[dict[str, Any]] = []
        self._lock = Lock()

    def record(self, trace_id: str, **metrics: Any) -> None:
        with self._lock:
            self._requests.setdefault(trace_id, {}).update(metrics)

    def record_stage(self, trace_id: str, stage: str, duration_ms: float, status: str) -> None:
        metric_name = self._STAGE_METRICS.get(stage)
        stage_key = f"{stage}_latency"
        payload: dict[str, Any] = {stage_key: duration_ms, f"{stage}_status": status}
        if metric_name:
            payload[metric_name] = duration_ms
        self.record(trace_id, **payload)

    def complete(self, trace_id: str, total_latency: float, status: str) -> None:
        with self._lock:
            metrics = dict(self._requests.pop(trace_id, {}))
            metrics.update({"trace_id": trace_id, "total_latency": total_latency, "status": status})
            self._completed.append(metrics)

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._completed[-1]) if self._completed else {}

    def summary(self) -> dict[str, Any]:
        with self._lock:
            completed = list(self._completed)
        if not completed:
            return {
                "requests_processed": 0,
                "average_latency": 0.0,
                "last_retrieval_score": 0.0,
                "last_confidence_score": 0.0,
            }
        average_latency = sum(float(item.get("total_latency", 0.0)) for item in completed) / len(completed)
        latest = completed[-1]
        return {
            "requests_processed": len(completed),
            "average_latency": average_latency,
            "last_retrieval_score": float(latest.get("average_similarity_score", 0.0)),
            "last_confidence_score": float(latest.get("confidence_score", 0.0)),
            "provider_name": latest.get("provider_name", ""),
        }

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._completed.clear()


_METRICS = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _METRICS
