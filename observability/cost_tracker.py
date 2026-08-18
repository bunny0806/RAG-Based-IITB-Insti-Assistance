"""Configurable token-cost estimates for supported LLM providers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping

from observability.metrics import MetricsCollector, get_metrics_collector


@dataclass(frozen=True)
class CostEstimate:
    input_cost: float
    output_cost: float
    total_cost: float


class CostTracker:
    """Estimate request cost using configurable USD-per-million-token prices."""

    DEFAULT_PRICING = {
        "gemini": {"input": 0.075, "output": 0.30},
        "groq": {"input": 0.59, "output": 0.79},
        "openai": {"input": 0.15, "output": 0.60},
        "anthropic": {"input": 0.80, "output": 4.00},
    }

    def __init__(self, pricing: Mapping[str, Mapping[str, float]] | None = None, metrics: MetricsCollector | None = None) -> None:
        selected_pricing = pricing or self._environment_pricing()
        self.pricing = {provider: dict(values) for provider, values in selected_pricing.items()}
        self.metrics = metrics or get_metrics_collector()

    @classmethod
    def _environment_pricing(cls) -> dict[str, dict[str, float]]:
        """Merge optional ``LLM_PRICING_JSON`` overrides into default provider prices."""
        pricing = {provider: dict(values) for provider, values in cls.DEFAULT_PRICING.items()}
        raw_overrides = os.getenv("LLM_PRICING_JSON", "").strip()
        if not raw_overrides:
            return pricing
        try:
            overrides = json.loads(raw_overrides)
        except json.JSONDecodeError:
            return pricing
        if not isinstance(overrides, dict):
            return pricing
        for provider, values in overrides.items():
            if isinstance(values, dict) and "input" in values and "output" in values:
                pricing[str(provider).lower()] = {
                    "input": float(values["input"]),
                    "output": float(values["output"]),
                }
        return pricing

    def set_pricing(self, provider: str, *, input_per_million: float, output_per_million: float) -> None:
        self.pricing[provider.lower()] = {"input": input_per_million, "output": output_per_million}

    def estimate(self, provider: str, input_tokens: int, output_tokens: int) -> CostEstimate:
        rates = self.pricing.get(provider.lower(), {"input": 0.0, "output": 0.0})
        input_cost = max(input_tokens, 0) * float(rates["input"]) / 1_000_000
        output_cost = max(output_tokens, 0) * float(rates["output"]) / 1_000_000
        return CostEstimate(input_cost=input_cost, output_cost=output_cost, total_cost=input_cost + output_cost)

    def record(self, trace_id: str, provider: str, input_tokens: int, output_tokens: int) -> CostEstimate:
        estimate = self.estimate(provider, input_tokens, output_tokens)
        self.metrics.record(
            trace_id,
            input_token_cost=estimate.input_cost,
            output_token_cost=estimate.output_cost,
            total_request_cost=estimate.total_cost,
        )
        return estimate


_COST_TRACKER = CostTracker()


def get_cost_tracker() -> CostTracker:
    return _COST_TRACKER
