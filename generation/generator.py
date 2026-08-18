"""Generation layer for turning retrieval results into grounded answers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

from prompts.prompt_builder import PromptBuilder
from rag.explainability import ExplainabilityReport
from retrieval.models import RetrievalResult
from utils.logging_utils import setup_logging

from llm.base import BaseLLM, FALLBACK_RESPONSE, LLMResponse
from llm.factory import LLMFactory
from observability import get_cost_tracker, get_current_trace, get_metrics_collector, trace_stage

logger = setup_logging("generation.log")


@dataclass(slots=True)
class GeneratedAnswer:
    """Structured output from the generation layer."""

    answer: str
    sources: List[str] = field(default_factory=list)
    used_chunks: List[str] = field(default_factory=list)
    latency: float = 0.0
    token_estimate: int = 0
    grounded: bool = False
    confidence_score: float = 0.0
    unsupported_claims: List[str] = field(default_factory=list)
    explainability_report: ExplainabilityReport | None = None
    # Memory-awareness fields
    original_query: Optional[str] = None
    resolved_query: Optional[str] = None
    summary_used: Optional[str] = None
    recent_messages_count: int = 0
    memory_hits: int = 0
    followup_detected: bool = False
    pronoun_resolved: bool = False
    memory_summary_length: int = 0
    recent_context_size: int = 0
    memory_retrieval_time_ms: float = 0.0
    provider_name: str = ""
    first_token_latency: float = 0.0
    token_count: int = 0


class GeneratedAnswerStream(Iterable[str]):
    """A response stream whose completed answer is available only after exhaustion."""

    def __init__(self, chunks: Iterator[str], on_complete: Callable[[], GeneratedAnswer]) -> None:
        self._chunks = chunks
        self._on_complete = on_complete
        self.final_answer: GeneratedAnswer | None = None

    def __iter__(self) -> Iterator[str]:
        completed = False
        try:
            yield from self._chunks
            completed = True
        finally:
            # A cancelled stream must never expose a partial response for storage.
            if completed:
                self.final_answer = self._on_complete()


class Generator:
    """Build prompts from retrieval results and generate grounded answers."""

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        gemini_client: Optional[BaseLLM] = None,
    ) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()
        # Keep the legacy keyword and attribute for existing callers, while the
        # generator itself depends exclusively on the provider-neutral contract.
        self.llm: BaseLLM = gemini_client or LLMFactory.create()
        self.gemini_client = self.llm

    def generate(self, question: str, retrieval_results: Sequence[RetrievalResult], context: dict | None = None) -> GeneratedAnswer:
        """Generate an answer using the supplied retrieval context."""
        start_time = time.perf_counter()
        logger.info("Generating answer for question: %s", question)
        logger.info("Generation provider: %s", self.llm.provider_name())
        logger.info("Retrieved %s chunk(s) for generation.", len(retrieval_results))

        prompt = self._build_prompt(question, retrieval_results, context)

        with trace_stage("generation"):
            try:
                response: LLMResponse = self.llm.generate(prompt)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.error("Generation failed: %s", exc)
                response = LLMResponse(
                    text=FALLBACK_RESPONSE,
                    model=self.llm.model_name,
                    latency_ms=0.0,
                    token_estimate=0,
                )

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Generation latency: %.2f ms", latency_ms)
        self._record_generation_metrics(prompt, response.text, response.token_estimate)

        return self._build_generated_answer(
            response.text,
            retrieval_results,
            context,
            latency_ms=latency_ms,
            token_estimate=response.token_estimate,
        )

    def stream(self, question: str, retrieval_results: Sequence[RetrievalResult], context: dict | None = None) -> GeneratedAnswerStream:
        """Stream response text while retaining a final ``GeneratedAnswer`` after completion."""
        prompt = self._build_prompt(question, retrieval_results, context)
        stream_start_time = time.perf_counter()
        chunks: list[str] = []
        first_token_latency_ms: float | None = None

        def _chunks() -> Iterator[str]:
            nonlocal first_token_latency_ms
            logger.info(
                "Streaming started | provider_name=%s | stream_start_time=%.6f",
                self.llm.provider_name(),
                stream_start_time,
            )
            with trace_stage("generation"):
                try:
                    for chunk in self.llm.stream(prompt):
                        if not chunk:
                            continue
                        if first_token_latency_ms is None:
                            first_token_latency_ms = (time.perf_counter() - stream_start_time) * 1000
                            logger.info("First token received | provider_name=%s | first_token_latency=%.2f ms", self.llm.provider_name(), first_token_latency_ms)
                        chunks.append(chunk)
                        yield chunk
                except Exception as exc:  # pragma: no cover - defensive path for custom providers
                    logger.error("Streaming generation failed: %s", exc)
                    if not chunks:
                        chunks.append(FALLBACK_RESPONSE)
                        yield FALLBACK_RESPONSE

            if not chunks:
                chunks.append(FALLBACK_RESPONSE)
                yield FALLBACK_RESPONSE

        def _complete() -> GeneratedAnswer:
            total_generation_time_ms = (time.perf_counter() - stream_start_time) * 1000
            response_text = "".join(chunks)
            token_count = len(response_text.split())
            first_latency = first_token_latency_ms if first_token_latency_ms is not None else total_generation_time_ms
            logger.info(
                "Streaming completed | provider_name=%s | total_generation_time=%.2f ms | first_token_latency=%.2f ms | token_count=%s",
                self.llm.provider_name(),
                total_generation_time_ms,
                first_latency,
                token_count,
            )
            self._record_generation_metrics(prompt, response_text, token_count, first_token_latency=first_latency)
            return self._build_generated_answer(
                response_text,
                retrieval_results,
                context,
                latency_ms=total_generation_time_ms,
                token_estimate=token_count,
                first_token_latency=first_latency,
                token_count=token_count,
            )

        return GeneratedAnswerStream(_chunks(), _complete)

    def stream_answer(self, question: str, retrieval_results: Sequence[RetrievalResult], context: dict | None = None) -> GeneratedAnswerStream:
        """Explicit alias for callers that prefer a descriptive streaming API."""
        return self.stream(question, retrieval_results, context)

    def _build_prompt(self, question: str, retrieval_results: Sequence[RetrievalResult], context: dict | None) -> str:
        """Build the single shared prompt for synchronous and streaming generation."""
        with trace_stage("prompt_building"):
            prompt = self.prompt_builder.build(question, retrieval_results, context=context)
        logger.info("Prompt size: %s characters", len(prompt))
        return prompt

    def _record_generation_metrics(
        self,
        prompt: str,
        response_text: str,
        output_tokens: int,
        *,
        first_token_latency: float | None = None,
    ) -> None:
        """Attach provider, token, latency, and estimated cost data to the active trace."""
        trace = get_current_trace()
        if trace is None:
            return
        input_tokens = len(prompt.split())
        metrics = {
            "token_usage": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "provider_name": self.llm.provider_name(),
            "model_name": self.llm.model_name,
        }
        if first_token_latency is not None:
            metrics["first_token_latency"] = first_token_latency
        get_metrics_collector().record(trace.trace_id, **metrics)
        get_cost_tracker().record(trace.trace_id, self.llm.provider_name(), input_tokens, output_tokens)

    def _build_generated_answer(
        self,
        response_text: str,
        retrieval_results: Sequence[RetrievalResult],
        context: dict | None,
        *,
        latency_ms: float,
        token_estimate: int,
        first_token_latency: float = 0.0,
        token_count: int = 0,
    ) -> GeneratedAnswer:
        """Create a common completed-answer object for both generation modes."""
        sources = self._collect_sources(retrieval_results)
        used_chunks = [result.chunk.chunk_id for result in retrieval_results if result.chunk.chunk_id]
        grounded = bool(retrieval_results and response_text and response_text != FALLBACK_RESPONSE)
        return GeneratedAnswer(
            answer=response_text,
            sources=sources,
            used_chunks=used_chunks,
            latency=latency_ms,
            token_estimate=token_estimate,
            grounded=grounded,
            original_query=(context.get("original_query") if context else None),
            resolved_query=(context.get("resolved_query") if context else None),
            summary_used=(context.get("summary") if context else None),
            recent_messages_count=(len(context.get("recent_messages")) if context and context.get("recent_messages") else 0),
            memory_hits=0,
            followup_detected=bool(context.get("followup_detected", False)) if context else False,
            pronoun_resolved=bool(context.get("pronoun_resolved", False)) if context else False,
            memory_summary_length=int(context.get("summary_length", 0)) if context else 0,
            recent_context_size=int(context.get("recent_context_size", 0)) if context else 0,
            memory_retrieval_time_ms=float(context.get("memory_retrieval_time_ms", 0.0)) if context else 0.0,
            provider_name=self.llm.provider_name(),
            first_token_latency=first_token_latency,
            token_count=token_count or token_estimate,
        )

    def _collect_sources(self, retrieval_results: Sequence[RetrievalResult]) -> List[str]:
        """Extract document names from retrieval results."""
        sources: List[str] = []
        for result in retrieval_results:
            chunk = result.chunk
            metadata = getattr(chunk, "metadata", {}) or {}
            document_name = metadata.get("document_name") or metadata.get("filename") or metadata.get("title")
            if document_name:
                sources.append(str(document_name))
            elif chunk.source:
                sources.append(str(chunk.source))
        return sources
