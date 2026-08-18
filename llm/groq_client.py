"""Groq implementation of the provider-neutral LLM interface."""

from __future__ import annotations

from typing import Any, Iterator, Optional

from config import GROQ_MODEL
from llm.base import BaseLLM
from utils.logging_utils import setup_logging

logger = setup_logging("generation.log")


class GroqClient(BaseLLM):
    """Groq adapter with a lazy optional SDK import."""

    def __init__(self, model_name: str = GROQ_MODEL) -> None:
        super().__init__(model_name=model_name, api_key_env="GROQ_API_KEY")
        self._client: Optional[Any] = None

    def provider_name(self) -> str:
        return "groq"

    def _generate_text(self, prompt: str) -> str:
        response = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _stream_text(self, prompt: str) -> Iterator[str]:
        responses = self._get_client().chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for response in responses:
            if response.choices:
                text = response.choices[0].delta.content
                if text:
                    yield text

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("Install the 'groq' package to use LLM_PROVIDER=groq.") from exc
            logger.info("Initializing Groq client for model %s", self.model_name)
            self._client = Groq(api_key=self._api_key())
        return self._client
