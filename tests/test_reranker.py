"""Unit tests for the CrossEncoder reranking layer."""

from __future__ import annotations

from typing import List

import pytest

from preprocessing.models import Chunk
from retrieval.models import RetrievalResult
from retrieval.reranker import Reranker
from retrieval.cross_encoder import CrossEncoderRanker


class DummyEncoder:
    def __init__(self, scores: List[float]) -> None:
        self._scores = scores

    def predict(self, pairs: List[tuple], batch_size: int = 32, show_progress_bar: bool = False) -> List[float]:
        return self._scores


@pytest.fixture(autouse=True)
def reset_cross_encoder_singleton() -> None:
    CrossEncoderRanker._instance = None


def test_reranker_sorts_candidates_by_cross_encoder_score(monkeypatch: pytest.MonkeyPatch) -> None:
    query = "best campus facilities"
    candidates = [
        RetrievalResult(chunk=Chunk(chunk_id="1", text="Campus cafeteria information.", metadata={}), score=0.0, rank=1),
        RetrievalResult(chunk=Chunk(chunk_id="2", text="Hostel fee details.", metadata={}), score=0.0, rank=2),
        RetrievalResult(chunk=Chunk(chunk_id="3", text="Admission process timeline.", metadata={}), score=0.0, rank=3),
    ]

    def fake_load_model(self):
        return DummyEncoder([0.1, 2.7, 1.3])

    monkeypatch.setattr(CrossEncoderRanker, "_load_model", fake_load_model)
    reranker = Reranker(candidate_k=3, reranked_k=3)
    reranked = reranker.rerank(query, candidates)

    assert [result.chunk.chunk_id for result in reranked] == ["2", "3", "1"]
    assert [result.score for result in reranked] == [2.7, 1.3, 0.1]


def test_reranker_limits_results_to_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    query = "hostel"
    candidates = [
        RetrievalResult(chunk=Chunk(chunk_id=str(i), text=f"Text {i}", metadata={}), score=0.0, rank=i)
        for i in range(1, 6)
    ]

    def fake_load_model(self):
        return DummyEncoder([float(i) for i in range(5)])

    monkeypatch.setattr(CrossEncoderRanker, "_load_model", fake_load_model)
    reranker = Reranker(candidate_k=5, reranked_k=2)
    reranked = reranker.rerank(query, candidates, top_k=2)

    assert len(reranked) == 2
    assert [result.chunk.chunk_id for result in reranked] == ["5", "4"]


def test_reranker_returns_empty_on_no_candidates() -> None:
    reranker = Reranker(candidate_k=20, reranked_k=5)
    assert reranker.rerank("query", []) == []


def test_cross_encoder_model_loads_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    query = "query"
    candidates = [RetrievalResult(chunk=Chunk(chunk_id="1", text="A", metadata={}), score=0.0, rank=1)]
    load_calls = {"count": 0}

    def fake_load_model(self):
        load_calls["count"] += 1
        return DummyEncoder([0.8])

    monkeypatch.setattr(CrossEncoderRanker, "_load_model", fake_load_model)
    reranker = Reranker(candidate_k=1, reranked_k=1)

    reranker.rerank(query, candidates)
    reranker.rerank(query, candidates)

    assert load_calls["count"] == 1


def test_reranker_raises_for_empty_query() -> None:
    reranker = Reranker(candidate_k=5, reranked_k=5)
    with pytest.raises(ValueError):
        reranker.rerank("", [RetrievalResult(chunk=Chunk(chunk_id="1", text="A", metadata={}), score=0.0, rank=1)])
