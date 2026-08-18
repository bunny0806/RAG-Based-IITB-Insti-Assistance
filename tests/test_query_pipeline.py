"""Unit tests for the intelligent query processing pipeline."""

from __future__ import annotations

from query.pipeline import ProcessedQuery, QueryPipeline
from query.classifier import QueryClassifier
from query.expander import QueryExpander
from query.rewriter import QueryRewriter


def test_query_expander_replaces_abbreviations() -> None:
    expander = QueryExpander()
    result = expander.expand("IITB SAC Hostel")

    assert "IIT Bombay" in result
    assert "Student Activity Centre" in result


def test_query_classifier_detects_hostel_category() -> None:
    classifier = QueryClassifier()
    category = classifier.classify("hostel fees and room allocation")

    assert category == "Hostel"


def test_query_rewriter_applies_rewrite_rule() -> None:
    rewriter = QueryRewriter()
    result = rewriter.rewrite("hostel fees")

    assert "hostel fee payment process" in result


def test_query_pipeline_processes_query_correctly() -> None:
    pipeline = QueryPipeline()
    processed = pipeline.process("wncc registration")

    assert isinstance(processed, ProcessedQuery)
    assert processed.original_query == "wncc registration"
    assert "Web and Coding Club" in processed.expanded_query
    assert processed.category in {"Clubs", "General"}
    assert processed.final_query != "wncc registration"


def test_query_pipeline_respects_disabled_features(monkeypatch) -> None:
    pipeline = QueryPipeline()

    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "False")
    monkeypatch.setenv("CLASSIFICATION_ENABLED", "False")
    monkeypatch.setenv("QUERY_REWRITING_ENABLED", "False")

    processed = pipeline.process("IITB registration")
    assert processed.expanded_query == "IITB registration"
    assert processed.rewritten_query == "IITB registration"
    assert processed.category == "General"


def test_query_pipeline_handles_empty_query() -> None:
    pipeline = QueryPipeline()
    try:
        pipeline.process("")
    except ValueError as exc:
        assert "non-empty string" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query")
