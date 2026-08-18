"""OpenAI implementation of the provider-neutral LLM interface."""

from __future__ import annotations

from typing import Any, Iterator, Optional

from config import OPENAI_MODEL
from llm.base import BaseLLM
from utils.logging_utils import setup_logging

logger = setup_logging("generation.log")


class OpenAIClient(BaseLLM):
    """OpenAI adapter with a lazy optional SDK import."""

    def __init__(self, model_name: str = OPENAI_MODEL) -> None:
        super().__init__(model_name=model_name, api_key_env="OPENAI_API_KEY")
        self._client: Optional[Any] = None

    def provider_name(self) -> str:
        return "openai"

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
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("Install the 'openai' package to use LLM_PROVIDER=openai.") from exc
            logger.info("Initializing OpenAI client for model %s", self.model_name)
            self._client = OpenAI(api_key=self._api_key())
        return self._client
