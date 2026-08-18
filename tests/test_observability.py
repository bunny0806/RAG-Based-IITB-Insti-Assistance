"""Tests for trace lifecycle, metrics, costs, failure handling, and generation integration."""

from __future__ import annotations

import pytest

from generation.generator import Generator
from llm.base import BaseLLM, LLMResponse
from observability.cost_tracker import CostTracker
from observability.metrics import MetricsCollector, get_metrics_collector
from observability.tracer import TraceContext


def test_trace_creation_records_request_identity_and_stage() -> None:
    with TraceContext(session_id="session-1") as trace:
        with trace.stage("retrieval"):
            pass

    assert trace.trace_id
    assert trace.timestamp
    assert trace.session_id == "session-1"
    assert trace.stages[0].stage == "retrieval"
    assert trace.stages[0].status == "success"
    assert trace.stages[0].duration >= 0


def test_metrics_recording_returns_health_summary() -> None:
    collector = MetricsCollector()
    collector.record("trace-1", average_similarity_score=0.82, confidence_score=0.91)
    collector.complete("trace-1", total_latency=125.0, status="success")

    summary = collector.summary()

    assert summary["requests_processed"] == 1
    assert summary["average_latency"] == 125.0
    assert summary["last_retrieval_score"] == 0.82
    assert summary["last_confidence_score"] == 0.91


def test_cost_tracker_calculates_configurable_prices() -> None:
    tracker = CostTracker(pricing={"openai": {"input": 2.0, "output": 4.0}})

    estimate = tracker.estimate("openai", input_tokens=1_000_000, output_tokens=500_000)

    assert estimate.input_cost == 2.0
    assert estimate.output_cost == 2.0
    assert estimate.total_cost == 4.0


def test_failed_stage_captures_error_message() -> None:
    with TraceContext(session_id="session-2") as trace:
        with pytest.raises(RuntimeError, match="retrieval failed"):
            with trace.stage("retrieval"):
                raise RuntimeError("retrieval failed")

    stage = trace.stages[0]
    assert stage.status == "failed"
    assert stage.error_message == "retrieval failed"
    assert stage.end_time is not None


class _ObservableLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(model_name="observable-model", api_key_env="OBSERVABLE_KEY")

    def provider_name(self) -> str:
        return "openai"

    def _generate_text(self, prompt: str) -> str:
        return "unused"

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text="Observable response", model=self.model_name, latency_ms=1.0, token_estimate=3)


def test_generator_records_provider_tokens_and_cost() -> None:
    metrics = get_metrics_collector()
    metrics.reset()
    generator = Generator(gemini_client=_ObservableLLM())

    with TraceContext(session_id="session-3"):
        generator.generate("What is IIT Bombay?", [])

    latest = metrics.latest()
    assert latest["provider_name"] == "openai"
    assert latest["model_name"] == "observable-model"
    assert latest["token_usage"] >= 3
    assert latest["total_request_cost"] > 0
