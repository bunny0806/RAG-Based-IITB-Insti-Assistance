"""Anthropic implementation of the provider-neutral LLM interface."""

from __future__ import annotations

from typing import Any, Iterator, Optional

from config import ANTHROPIC_MODEL
from llm.base import BaseLLM
from utils.logging_utils import setup_logging

logger = setup_logging("generation.log")


class AnthropicClient(BaseLLM):
    """Anthropic adapter with a lazy optional SDK import."""

    def __init__(self, model_name: str = ANTHROPIC_MODEL) -> None:
        super().__init__(model_name=model_name, api_key_env="ANTHROPIC_API_KEY")
        self._client: Optional[Any] = None

    def provider_name(self) -> str:
        return "anthropic"

    def _generate_text(self, prompt: str) -> str:
        response = self._get_client().messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")

    def _stream_text(self, prompt: str) -> Iterator[str]:
        with self._get_client().messages.stream(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as response_stream:
            yield from response_stream.text_stream

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("Install the 'anthropic' package to use LLM_PROVIDER=anthropic.") from exc
            logger.info("Initializing Anthropic client for model %s", self.model_name)
            self._client = Anthropic(api_key=self._api_key())
        return self._client
