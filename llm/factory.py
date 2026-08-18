"""Factory for selecting an LLM provider from configuration."""

from __future__ import annotations

import os
from typing import Type

from config import LLM_PROVIDER
from llm.anthropic_client import AnthropicClient
from llm.base import BaseLLM
from llm.gemini_client import GeminiClient
from llm.groq_client import GroqClient
from llm.openai_client import OpenAIClient
from utils.logging_utils import setup_logging

logger = setup_logging("generation.log")


class LLMFactory:
    """Construct configured provider adapters without exposing provider logic to callers."""

    _providers: dict[str, Type[BaseLLM]] = {
        "gemini": GeminiClient,
        "groq": GroqClient,
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }

    @classmethod
    def create(cls, provider: str | None = None) -> BaseLLM:
        """Create the provider selected by ``LLM_PROVIDER`` or an explicit test override."""
        selected_provider = (provider or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).strip().lower()
        client_class = cls._providers.get(selected_provider)
        if client_class is None:
            supported = ", ".join(sorted(cls._providers))
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{selected_provider}'. Supported providers: {supported}."
            )
        client = client_class()
        logger.info("LLM provider selected: %s (model=%s)", client.provider_name(), client.model_name)
        return client
