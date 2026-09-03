from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ordered from most to least semantically meaningful. "" means "split on every
# character" and is the final fallback for pathological inputs (e.g. a 10k-char
# string with no whitespace).
DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A single retrievable unit of text plus provenance metadata."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _split_keep(text: str, separator: str) -> List[str]:
    """Split ``text`` on ``separator`` but re-attach the separator to the
    preceding piece so no characters are silently dropped."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    out: List[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            out.append(part + separator)
        else:
            out.append(part)
    return [p for p in out if p != ""]


def _recursive_split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """Break ``text`` into pieces each <= ``chunk_size`` where possible, always
    preferring the earliest (most meaningful) separator that appears."""
    if not text:
        return []

    separator = separators[-1]
    remaining = separators[-1:]
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    pieces = _split_keep(text, separator)
    final: List[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            if piece:
                final.append(piece)
        elif remaining:
            final.extend(_recursive_split(piece, remaining, chunk_size))
        else:
            # No separators left: hard-cut the oversized piece.
            for j in range(0, len(piece), chunk_size):
                final.append(piece[j : j + chunk_size])
    return final


def _merge(splits: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Greedily pack ``splits`` into chunks up to ``chunk_size`` chars, carrying
    an ``overlap``-char tail from each chunk into the next."""
    chunks: List[str] = []
    current = ""
    for piece in splits:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if overlap > 0 else ""
        current += piece
    if current.strip():
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Split raw ``text`` into overlapping character chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    separators = separators or DEFAULT_SEPARATORS
    splits = _recursive_split(text, separators, chunk_size)
    return _merge(splits, chunk_size, chunk_overlap)


def chunk_document(
    text: str,
    source: str,
    page: Optional[int] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    """Split a document into :class:`Chunk` objects with stable ids + metadata.

    The ``chunk_id`` is deterministic (``{source}::p{page}::c{index}``) so the
    same chunk id can be referenced by both the vector store and the knowledge
    graph - that shared id is what makes end-to-end citations work.
    """
    chunks: List[Chunk] = []
    for idx, piece in enumerate(split_text(text, chunk_size, chunk_overlap)):
        page_tag = f"p{page}" if page is not None else "p0"
        chunk_id = f"{source}::{page_tag}::c{idx}"
        chunks.append(
            Chunk(
                text=piece,
                metadata={
                    "chunk_id": chunk_id,
                    "source": source,
                    "page": page,
                    "index": idx,
                },
            )
        )
    return chunks
