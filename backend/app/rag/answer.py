from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence


def build_context_block(contexts: Sequence[Dict[str, Any]]) -> str:
    """Render retrieved chunks into a numbered context block.

    Each context is prefixed with an index like ``[1]`` plus its source/page so
    the model can cite with ``[1]`` and the UI can map that citation back to the
    exact document and page.
    """
    lines: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        meta = ctx.get("metadata", {}) or {}
        src = ctx.get("source") or meta.get("source") or "unknown"
        page = ctx.get("page")
        if page is None:
            page = meta.get("page")
        loc = str(src) + (f" p.{page}" if page is not None else "")
        text = (ctx.get("text") or "").strip()
        lines.append(f"[{i}] (source: {loc})\n{text}")
    return "\n\n".join(lines)


class AnswerGenerator:
    """Turns a query + retrieved contexts into a grounded, cited answer."""

    def __init__(self, llm: Any):
        self.llm = llm

    def _messages(self, query: str, contexts: Sequence[Dict[str, Any]]):
        from app.llm.prompts import QA_SYSTEM, qa_user_prompt

        block = build_context_block(contexts)
        return [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": qa_user_prompt(query, block)},
        ]

    def generate(self, query: str, contexts: Sequence[Dict[str, Any]]) -> str:
        return self.llm.chat(self._messages(query, contexts), temperature=0.1)

    def stream(self, query: str, contexts: Sequence[Dict[str, Any]]) -> Iterator[str]:
        yield from self.llm.stream_chat(self._messages(query, contexts), temperature=0.1)
