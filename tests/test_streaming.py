"""Tests for provider streaming, completed-answer handling, and UI consumption."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from generation.generator import GeneratedAnswer, Generator
from llm.base import BaseLLM, LLMResponse
from memory.memory_manager import MemoryManager
from preprocessing.models import Chunk
from retrieval.models import RetrievalResult
from ui.app import (
    _finalize_conversational_turn,
    _prepare_conversational_turn,
    _render_streaming_answer,
)


class _ChunkingLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(model_name="streaming-test", api_key_env="STREAMING_TEST_KEY")
        self.prompts: list[str] = []

    def provider_name(self) -> str:
        return "streaming-test"

    def _generate_text(self, prompt: str) -> str:
        return "fallback response"

    def stream(self, prompt: str):
        self.prompts.append(prompt)
        yield "Final "
        yield "response."


class _FallbackLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(model_name="fallback-test", api_key_env="FALLBACK_TEST_KEY")

    def provider_name(self) -> str:
        return "fallback-test"

    def _generate_text(self, prompt: str) -> str:
        return "complete fallback"


def test_base_llm_streaming_interface_yields_chunks() -> None:
    llm = _ChunkingLLM()

    assert list(llm.stream("Prompt")) == ["Final ", "response."]


def test_base_llm_stream_falls_back_to_generate(monkeypatch) -> None:
    monkeypatch.setenv("FALLBACK_TEST_KEY", "configured")
    llm = _FallbackLLM()

    assert list(llm.stream("Prompt")) == ["complete fallback"]


def test_generator_streams_with_same_prompt_builder() -> None:
    llm = _ChunkingLLM()
    generator = Generator(gemini_client=llm)

    answer_stream = generator.stream("What is IIT Bombay?", [])

    assert "".join(answer_stream) == "Final response."
    assert answer_stream.final_answer is not None
    assert answer_stream.final_answer.answer == "Final response."
    assert answer_stream.final_answer.provider_name == "streaming-test"
    assert answer_stream.final_answer.first_token_latency >= 0
    assert llm.prompts and "Current User Query:" in llm.prompts[0]


class _FakeRetrievalPipeline:
    def retrieve(self, query: str, top_k: int | None = None):
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="stream-chunk",
                    text="IIT Bombay streaming test context.",
                    metadata={"document_id": "stream-doc", "document_name": "stream.pdf"},
                    document_id="stream-doc",
                    source="stream.pdf",
                ),
                score=0.9,
                rank=1,
            )
        ]


@dataclass
class _EvaluationResult:
    confidence_score: float = 0.9
    unsupported_claims: list[str] | None = None


def test_memory_stores_only_completed_stream_response(monkeypatch, tmp_path) -> None:
    manager = MemoryManager(storage_dir=str(tmp_path / "memory"))
    generator = Generator(gemini_client=_ChunkingLLM())
    original_query = "Tell me about IIT Bombay"
    resolved_query, retrieval_results, context = _prepare_conversational_turn(
        manager,
        "stream-session",
        original_query,
        _FakeRetrievalPipeline(),
    )
    answer_stream = generator.stream(resolved_query, retrieval_results, context)
    iterator = iter(answer_stream)

    assert next(iterator) == "Final "
    assert manager.get_memory("stream-session").length() == 0

    assert "".join(iterator) == "response."
    assert answer_stream.final_answer is not None
    monkeypatch.setattr(
        "ui.app.ResponseValidator.validate",
        lambda self, answer, results: _EvaluationResult(unsupported_claims=[]),
    )
    _finalize_conversational_turn(
        manager,
        "stream-session",
        original_query,
        resolved_query,
        answer_stream.final_answer,
        retrieval_results,
        context,
    )

    entries = manager.get_memory("stream-session").list_entries()
    assert len(entries) == 1
    assert entries[0].assistant_response == "Final response."


class _Placeholder:
    def __init__(self) -> None:
        self.cleared = False

    @contextmanager
    def container(self):
        yield self

    def empty(self) -> None:
        self.cleared = True


class _FakeStreamlit:
    def __init__(self) -> None:
        self.placeholder = _Placeholder()
        self.rendered = ""

    def empty(self) -> _Placeholder:
        return self.placeholder

    def write_stream(self, stream) -> str:
        self.rendered = "".join(stream)
        return self.rendered


class _UiCompatibleStream:
    def __init__(self) -> None:
        self.final_answer: GeneratedAnswer | None = None

    def __iter__(self):
        yield "Visible "
        yield "typing"
        self.final_answer = GeneratedAnswer(answer="Visible typing")


def test_streaming_ui_consumes_stream_and_uses_completed_answer(monkeypatch) -> None:
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setattr("ui.app.st", fake_streamlit)
    answer_stream = _UiCompatibleStream()

    answer = _render_streaming_answer(answer_stream)

    assert fake_streamlit.rendered == "Visible typing"
    assert fake_streamlit.placeholder.cleared is True
    assert answer.answer == "Visible typing"
