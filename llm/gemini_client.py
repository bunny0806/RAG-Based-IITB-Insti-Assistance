"""Gemini implementation of the provider-neutral LLM interface."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import google.generativeai as genai

from config import GEMINI_MODEL
from llm.base import BaseLLM, LLMResponse
from utils.logging_utils import setup_logging

logger = setup_logging("generation.log")

# Kept as an alias so existing imports of GeminiResponse remain valid.
GeminiResponse = LLMResponse


class GeminiClient(BaseLLM):
    """Gemini adapter with lazy SDK initialization."""

    def __init__(self, model_name: str = GEMINI_MODEL) -> None:
        super().__init__(model_name=model_name, api_key_env="GEMINI_API_KEY")
        self._model: Optional[Any] = None

    def provider_name(self) -> str:
        return "gemini"

    def _generate_text(self, prompt: str) -> str:
        response = self._get_model().generate_content(prompt, timeout=30)
        return self._extract_text(response)

    def _stream_text(self, prompt: str) -> Iterator[str]:
        responses = self._get_model().generate_content(prompt, stream=True)
        for response in responses:
            text = self._extract_text(response)
            if text:
                yield text

    def _get_model(self) -> Any:
        if self._model is None:
            logger.info("Initializing Gemini client for model %s", self.model_name)
            genai.configure(api_key=self._api_key())
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def _extract_text(self, response: Any) -> str:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text
        return ""
