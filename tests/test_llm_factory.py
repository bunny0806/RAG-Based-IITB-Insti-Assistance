"""Tests for provider-neutral LLM selection and generator dependency inversion."""

from __future__ import annotations

import pytest

from generation.generator import Generator
from llm.anthropic_client import AnthropicClient
from llm.base import BaseLLM, LLMResponse
from llm.factory import LLMFactory
from llm.gemini_client import GeminiClient
from llm.groq_client import GroqClient
from llm.openai_client import OpenAIClient


@pytest.mark.parametrize(
    ("provider", "client_type"),
    [
        ("gemini", GeminiClient),
        ("groq", GroqClient),
        ("openai", OpenAIClient),
        ("anthropic", AnthropicClient),
    ],
)
def test_factory_selects_provider_from_environment(monkeypatch, provider, client_type) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider)

    client = LLMFactory.create()

    assert isinstance(client, client_type)
    assert client.provider_name() == provider


def test_factory_switches_provider_without_code_changes(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    first_client = LLMFactory.create()
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    second_client = LLMFactory.create()

    assert first_client.provider_name() == "gemini"
    assert second_client.provider_name() == "openai"


def test_factory_rejects_unsupported_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unsupported-provider")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER 'unsupported-provider'"):
        LLMFactory.create()


class _FakeLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(model_name="fake-model", api_key_env="FAKE_API_KEY")
        self.prompts: list[str] = []

    def provider_name(self) -> str:
        return "fake"

    def _generate_text(self, prompt: str) -> str:
        return "unused"

    def generate(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(text="Provider-neutral answer", model=self.model_name, latency_ms=1.0, token_estimate=4)


def test_generator_uses_base_llm_contract() -> None:
    llm = _FakeLLM()
    # The legacy keyword remains supported, but accepts the provider-neutral type.
    generator = Generator(gemini_client=llm)

    answer = generator.generate("What is IIT Bombay?", [])

    assert generator.llm is llm
    assert generator.gemini_client is llm
    assert answer.answer == "Provider-neutral answer"
    assert llm.prompts
