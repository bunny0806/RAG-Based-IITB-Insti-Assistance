"""Provider-neutral contracts and shared behavior for LLM clients."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from utils.logging_utils import setup_logging

logger = setup_logging("generation.log")

FALLBACK_RESPONSE = "I don't know based on the available IIT Bombay documents."


@dataclass(slots=True)
class LLMResponse:
    """Provider-neutral text-generation result."""

    text: str
    model: str
    latency_ms: float
    token_estimate: int


class BaseLLM(ABC):
    """Common interface and resilience behavior for supported LLM providers."""

    def __init__(self, model_name: str, api_key_env: str) -> None:
        self.model_name = model_name
        self.api_key_env = api_key_env

    def generate(self, prompt: str) -> LLMResponse:
        """Generate text with consistent validation, retries, and fallback behavior."""
        self._validate_prompt(prompt)
        if not self._api_key():
            logger.warning("%s is not configured. Returning fallback response.", self.api_key_env)
            return self._fallback_response()

        start_time = time.perf_counter()
        try:
            response_text = self._run_with_retries(lambda: self._generate_text(prompt))
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info("%s generation completed in %.2f ms", self.provider_name(), latency_ms)
            return LLMResponse(
                text=response_text,
                model=self.model_name,
                latency_ms=latency_ms,
                token_estimate=self._estimate_tokens(prompt, response_text),
            )
        except Exception as exc:  # pragma: no cover - defensive provider path
            logger.error("%s generation failed: %s", self.provider_name(), exc)
            return self._fallback_response((time.perf_counter() - start_time) * 1000)

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield provider chunks, falling back safely to one generated response."""
        self._validate_prompt(prompt)
        if not self._api_key():
            yield self.generate(prompt).text
            return

        yielded_chunk = False
        try:
            for chunk in self._stream_text(prompt):
                if chunk:
                    yielded_chunk = True
                    yield chunk
            if not yielded_chunk:
                logger.warning("%s returned no streaming chunks; using synchronous fallback.", self.provider_name())
                yield self.generate(prompt).text
        except NotImplementedError:
            logger.info("%s has no native streaming implementation; using synchronous fallback.", self.provider_name())
            yield self.generate(prompt).text
        except Exception as exc:  # pragma: no cover - provider/network dependent
            logger.error("%s streaming failed: %s", self.provider_name(), exc)
            # Avoid appending a full second response after partial output.
            if not yielded_chunk:
                yield self.generate(prompt).text

    def health_check(self) -> bool:
        """Report whether the provider has the configuration needed to serve calls."""
        return bool(self._api_key())

    @abstractmethod
    def provider_name(self) -> str:
        """Return the stable provider identifier used by diagnostics."""

    @abstractmethod
    def _generate_text(self, prompt: str) -> str:
        """Perform the provider-specific request and return its text."""

    def _stream_text(self, prompt: str) -> Iterator[str]:
        """Perform native provider streaming when supported by the SDK."""
        raise NotImplementedError

    def _api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)

    def _run_with_retries(self, operation: Callable[[], str], retries: int = 3) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return operation()
            except Exception as exc:  # pragma: no cover - depends on provider failures
                last_error = exc
                logger.warning("%s attempt %s failed: %s", self.provider_name(), attempt + 1, exc)
                if attempt < retries - 1:
                    time.sleep(1)
        raise RuntimeError(f"{self.provider_name()} generation failed after retries") from last_error

    def _validate_prompt(self, prompt: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

    def _fallback_response(self, latency_ms: float = 0.0) -> LLMResponse:
        return LLMResponse(
            text=FALLBACK_RESPONSE,
            model=self.model_name,
            latency_ms=latency_ms,
            token_estimate=0,
        )

    def _estimate_tokens(self, prompt: str, response_text: str) -> int:
        """Estimate tokens consistently when providers omit usage metadata."""
        return len(prompt.split()) + len(response_text.split())
