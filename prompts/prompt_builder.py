"""Build structured prompts for the generation layer."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from retrieval.models import RetrievalResult

from .system_prompt import SYSTEM_PROMPT


class PromptBuilder:
    """Construct a prompt that includes system instructions, retrieved context, and citations."""

    def build(self, question: str, retrieval_results: Sequence[RetrievalResult], context: dict | None = None) -> str:
        """Create a structured prompt for generation."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        # context may include conversation summary and recent messages
        convo_summary = context.get("summary") if context else None
        recent = context.get("recent_messages") if context else None

        context_blocks = self._build_context_blocks(retrieval_results)
        sources = self._build_sources(retrieval_results)

        prompt_parts = [f"System Prompt:\n{SYSTEM_PROMPT}", ""]

        if convo_summary:
            prompt_parts.extend(["Conversation Summary:", convo_summary, ""])

        if recent:
            prompt_parts.extend(["Recent Conversation:", "\n".join(recent), ""])

        prompt_parts.extend(["Retrieved Context:", context_blocks, "", "Sources:", sources, ""]) 

        prompt_parts.extend([
            "Current User Query:",
            question,
            "",
            "Instructions:",
            "- Use retrieved context as the source of factual claims.",
            "- Use conversation context only to resolve references in the current query.",
            "- Cite the document names used.",
            "- If the context is insufficient, respond exactly: 'I don't know based on the available IIT Bombay documents.'",
            "- Be concise and use markdown formatting.",
        ])
        return "\n".join(prompt_parts)

    def _build_context_blocks(self, retrieval_results: Sequence[RetrievalResult]) -> str:
        """Render retrieved chunks into a context section."""
        if not retrieval_results:
            return "No relevant context was retrieved."

        blocks: List[str] = []
        for result in retrieval_results:
            chunk = result.chunk
            block = [
                f"Document: {self._get_document_name(chunk)}",
                f"Score: {result.score:.4f}",
                chunk.text,
            ]
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)

    def _build_sources(self, retrieval_results: Sequence[RetrievalResult]) -> str:
        """List document names used as sources."""
        if not retrieval_results:
            return "No sources available."

        sources = [self._get_document_name(result.chunk) for result in retrieval_results]
        return "\n".join(f"- {source}" for source in sources)

    def _get_document_name(self, chunk: Any) -> str:
        """Extract a readable document name from chunk metadata or source."""
        metadata = getattr(chunk, "metadata", {}) or {}
        document_name = metadata.get("document_name") or metadata.get("filename") or metadata.get("title")
        if document_name:
            return str(document_name)
        source = getattr(chunk, "source", "") or "unknown"
        return str(source)
