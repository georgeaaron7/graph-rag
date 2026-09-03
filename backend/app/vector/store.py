"""FAISS-backed vector store with on-disk persistence.

Embeddings are L2-normalized and searched with inner product, which equals
cosine similarity for normalized vectors. Chunk text + metadata live in a JSON
sidecar keyed by the same ``chunk_id`` used in the knowledge graph, so a vector
hit and a graph hit refer to the *identical* unit of text - that shared id is
what lets the two retrievers be fused and cited consistently.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

from app.retrieval.hybrid import ScoredChunk


class VectorStore:
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunk_ids: List[str] = []
        self._meta: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def add(
        self,
        chunk_ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        records: Sequence[Dict[str, Any]],
    ) -> None:
        import numpy as np

        self._index.add(np.asarray(vectors, dtype="float32"))
        for cid, rec in zip(chunk_ids, records):
            self._chunk_ids.append(cid)
            self._meta[cid] = rec

    def search(self, query_vector: Sequence[float], top_k: int = 8) -> List[ScoredChunk]:
        import numpy as np

        if not self._chunk_ids:
            return []
        q = np.asarray([query_vector], dtype="float32")
        scores, idxs = self._index.search(q, min(top_k, len(self._chunk_ids)))
        hits: List[ScoredChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._chunk_ids):
                continue
            cid = self._chunk_ids[idx]
            rec = self._meta.get(cid, {})
            hits.append(
                ScoredChunk(
                    chunk_id=cid,
                    text=rec.get("text", ""),
                    score=float(score),
                    source="vector",
                    metadata=rec,
                )
            )
        return hits

    def save(self, directory: str) -> None:
        import faiss

        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "store.json"), "w") as f:
            json.dump(
                {"chunk_ids": self._chunk_ids, "meta": self._meta, "dim": self.dim}, f
            )

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        import faiss

        with open(os.path.join(directory, "store.json")) as f:
            data = json.load(f)
        store = cls.__new__(cls)  # bypass __init__ (index comes from disk)
        store.dim = data["dim"]
        store._index = faiss.read_index(os.path.join(directory, "index.faiss"))
        store._chunk_ids = data["chunk_ids"]
        store._meta = data["meta"]
        return store
