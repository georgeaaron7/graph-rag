"""hybrid retrieval: fuse dense vector hits with knowledge-graph hits.
the two retrievers answer different questions:
* the vector store finds chunks that are *semantically* similar to the query;
* the graph store finds chunks attached to entities that are *structurally*
  connected to the query's entities (multi-hop) - which plain vector search
  misses entirely.

combine their rankings with reciprocal rank fusion. RRF is
rank-based, so we never have to reconcile a cosine similarity with a graph
hop-count on the same numeric scale - each retriever just contributes an
ordering, and an item ranked highly by *either* retriever bubbles up. That
property is exactly why GraphRAG beats vector-only RAG on multi-hop questions,
and it is a clean, honest thing to explain in an interview.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class ScoredChunk:
    """A retrieved unit of text with a score and which retriever produced it."""

    chunk_id: str
    text: str = ""
    score: float = 0.0
    source: str = "vector"  # "vector" | "graph" | "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> Dict[str, float]:
    """Return ``{chunk_id: fused_score}`` from several ranked id lists.

    ``score = sum over lists of 1 / (k + rank)`` with ``rank`` starting at 1.
    ``k`` damps the influence of very high ranks; 60 is the value from the
    original RRF paper (Cormack et al.) and a sane default.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse(
    vector_hits: Sequence[ScoredChunk],
    graph_hits: Sequence[ScoredChunk],
    k: int = 60,
    top_k: int = 8,
) -> List[ScoredChunk]:
    """Fuse two ranked hit lists into a single ranked list via RRF.

    The returned chunks are tagged ``hybrid`` when they were found by *both*
    retrievers, otherwise ``vector`` / ``graph`` - useful for showing in the UI
    *why* a source was retrieved.
    """
    richest: Dict[str, ScoredChunk] = {}
    for hit in list(vector_hits) + list(graph_hits):
        existing = richest.get(hit.chunk_id)
        # keep whichever record actually carries the chunk text
        if existing is None or (not existing.text and hit.text):
            richest[hit.chunk_id] = hit

    fused_scores = reciprocal_rank_fusion(
        [[h.chunk_id for h in vector_hits], [h.chunk_id for h in graph_hits]],
        k=k,
    )
    vector_ids = {h.chunk_id for h in vector_hits}
    graph_ids = {h.chunk_id for h in graph_hits}

    results: List[ScoredChunk] = []
    for chunk_id, score in fused_scores.items():
        base = richest[chunk_id]
        if chunk_id in vector_ids and chunk_id in graph_ids:
            src = "hybrid"
        elif chunk_id in graph_ids:
            src = "graph"
        else:
            src = "vector"
        results.append(
            ScoredChunk(
                chunk_id=chunk_id,
                text=base.text,
                score=score,
                source=src,
                metadata=base.metadata,
            )
        )
    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_k]
