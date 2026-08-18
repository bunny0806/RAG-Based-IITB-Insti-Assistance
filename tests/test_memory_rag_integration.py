from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from generation.generator import GeneratedAnswer
from memory.conversation_memory import ConversationEntry
from memory.memory_manager import MemoryManager
from preprocessing.models import Chunk
from retrieval.models import RetrievalResult
from ui.app import _process_conversational_turn
from prompts.prompt_builder import PromptBuilder


@dataclass
class _FakeEvaluationResult:
    grounded: bool = True
    confidence_score: float = 0.9
    reason: str = "ok"
    retrieved_sources: list[str] | None = None
    unsupported_claims: list[str] | None = None


class _FakeRetrievalPipeline:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int | None = None):
        self.queries.append(query)
        if "CS101" in query:
            return [
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id="chunk-cs101",
                        text="CS101 grading policy uses a weighted distribution.",
                        metadata={"document_id": "doc-cs101", "filename": "cs101.pdf", "document_name": "cs101.pdf"},
                        document_id="doc-cs101",
                        source="cs101.pdf",
                    ),
                    score=0.1,
                    rank=1,
                    retrieval_method="hybrid",
                    metadata={"document_id": "doc-cs101"},
                )
            ]
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="chunk-general",
                    text="General campus policy information.",
                    metadata={"document_id": "doc-general", "filename": "general.pdf", "document_name": "general.pdf"},
                    document_id="doc-general",
                    source="general.pdf",
                ),
                score=0.5,
                rank=1,
                retrieval_method="hybrid",
                metadata={"document_id": "doc-general"},
            )
        ]


class _FakeGenerator:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, question, retrieval_results, context=None):
        prompt = PromptBuilder().build(question, retrieval_results, context=context)
        self.prompts.append(prompt)
        return GeneratedAnswer(
            answer="Grounded answer about CS101.",
            sources=["cs101.pdf"],
            used_chunks=[result.chunk.chunk_id for result in retrieval_results],
            grounded=True,
            confidence_score=0.9,
        )


def _fake_validate(self, generated_answer, retrieval_results):
    return _FakeEvaluationResult(retrieved_sources=["cs101.pdf"], unsupported_claims=[])


def test_retrieval_uses_resolved_query_and_updates_memory(tmp_path: Path, monkeypatch) -> None:
    manager = MemoryManager(storage_dir=str(tmp_path / "memory"))
    session_id = "session-a"
    memory = manager.get_memory(session_id)
    memory.add_entry(
        ConversationEntry(user_query="Tell me about CS101", assistant_response="CS101 is a course about programming.")
    )

    retrieval_pipeline = _FakeRetrievalPipeline()
    generator = _FakeGenerator()

    monkeypatch.setattr("ui.app.ResponseValidator.validate", _fake_validate)

    answer, retrieval_results, context = _process_conversational_turn(
        manager,
        session_id,
        "What about its grading?",
        retrieval_pipeline,
        generator,
    )

    assert "CS101" in retrieval_pipeline.queries[0]
    assert "its" not in retrieval_pipeline.queries[0].lower()
    assert retrieval_results[0].chunk.document_id == "doc-cs101"
    assert answer.resolved_query is not None
    assert answer.followup_detected is True
    assert answer.pronoun_resolved is True
    assert memory.length() == 2
    assert context["summary"] == ""


def test_prompt_builder_injects_memory_context(tmp_path: Path, monkeypatch) -> None:
    manager = MemoryManager(storage_dir=str(tmp_path / "memory2"))
    session_id = "session-b"
    memory = manager.get_memory(session_id)
    for index in range(6):
        memory.add_entry(
            ConversationEntry(
                user_query=f"Tell me about CS101 topic {index}",
                assistant_response="CS101 is a foundational course.",
            )
        )

    retrieval_pipeline = _FakeRetrievalPipeline()
    generator = _FakeGenerator()
    monkeypatch.setattr("ui.app.ResponseValidator.validate", _fake_validate)

    _process_conversational_turn(
        manager,
        session_id,
        "What about its grading?",
        retrieval_pipeline,
        generator,
    )

    prompt = generator.prompts[-1]
    assert "Conversation Summary:" in prompt
    assert "CS101" in prompt
    assert "Recent Conversation:" in prompt
    assert "Retrieved Context:" in prompt
    assert "Current User Query:" in prompt
    assert prompt.index("System Prompt:") < prompt.index("Conversation Summary:")
    assert prompt.index("Conversation Summary:") < prompt.index("Recent Conversation:")
    assert prompt.index("Recent Conversation:") < prompt.index("Retrieved Context:")
    assert prompt.index("Retrieved Context:") < prompt.index("Current User Query:")


def test_memory_is_updated_after_every_response(tmp_path: Path, monkeypatch) -> None:
    manager = MemoryManager(storage_dir=str(tmp_path / "memory3"))
    session_id = "session-c"
    retrieval_pipeline = _FakeRetrievalPipeline()
    generator = _FakeGenerator()
    monkeypatch.setattr("ui.app.ResponseValidator.validate", _fake_validate)

    _process_conversational_turn(manager, session_id, "Tell me about CS101", retrieval_pipeline, generator)
    _process_conversational_turn(manager, session_id, "What about its grading?", retrieval_pipeline, generator)

    memory = manager.get_memory(session_id)
    assert memory.length() == 2
    entries = memory.list_entries()
    assert entries[0].original_query == "Tell me about CS101"
    assert entries[1].resolved_query is not None
    assert entries[1].retrieved_document_ids == ["doc-cs101"]
    assert entries[1].citations == ["cs101.pdf"]
    assert entries[1].confidence == 0.9
    assert entries[1].timestamp


def test_followup_without_pronoun_keeps_diagnostics_distinct(tmp_path: Path, monkeypatch) -> None:
    manager = MemoryManager(storage_dir=str(tmp_path / "memory4"))
    session_id = "session-d"
    manager.get_memory(session_id).add_entry(
        ConversationEntry(user_query="Tell me about CS101", assistant_response="CS101 is a course.")
    )
    retrieval_pipeline = _FakeRetrievalPipeline()
    generator = _FakeGenerator()
    monkeypatch.setattr("ui.app.ResponseValidator.validate", _fake_validate)

    answer, _, context = _process_conversational_turn(
        manager,
        session_id,
        "What about grading?",
        retrieval_pipeline,
        generator,
    )

    assert context["followup_detected"] is True
    assert context["pronoun_resolved"] is False
    assert answer.followup_detected is True
    assert answer.pronoun_resolved is False
