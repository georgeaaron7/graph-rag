from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

from app.ingestion.chunking import Chunk, chunk_document
from app.ingestion.pdf_loader import load_pdf
from app.graph.extraction import extract_graph_chunk, parse_extraction
from app.rag.answer import AnswerGenerator
from app.retrieval.hybrid import ScoredChunk, fuse


class IngestionPipeline:
    """PDF -> chunks -> (vectors in FAISS) + (entities/relations in Neo4j)."""

    def __init__(
        self,
        embedder,
        vector_store_factory: Callable[[int], Any],
        graph_store,
        llm,
        settings,
    ):
        self.embedder = embedder
        self.vector_store_factory = vector_store_factory  # callable(dim) -> VectorStore
        self.graph_store = graph_store
        self.llm = llm
        self.settings = settings
        self.vector_store = None

    def ingest_pdf(self, path: str) -> Dict[str, int]:
        source = os.path.basename(path)
        chunks: List[Chunk] = []
        for page in load_pdf(path):
            if not page.text.strip():
                continue
            chunks.extend(
                chunk_document(
                    page.text,
                    source=source,
                    page=page.page_number,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
            )
        return self._index_chunks(chunks, source)

    def ingest_text(self, text: str, source: str, page: int = 1) -> Dict[str, int]:
        """Ingest a raw-text document through the same chunk -> vector + graph
        path as a PDF. Useful for text corpora / benchmarks (e.g. MultiHop-RAG),
        where ``source`` becomes the chunk_id prefix (``{source}::p{page}::c{i}``)."""
        chunks = chunk_document(
            text,
            source=source,
            page=page,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        return self._index_chunks(chunks, source)

    def _index_chunks(self, chunks: List[Chunk], source: str) -> Dict[str, int]:
        """Shared indexing: embed + store vectors, then extract + upsert graph."""
        if not chunks:
            return {"chunks": 0, "entities": 0, "relations": 0}

        # --- vectors ---
        vectors = self.embedder.embed_texts([c.text for c in chunks])
        if self.vector_store is None:
            self.vector_store = self.vector_store_factory(self.embedder.dimension)
        self.vector_store.add(
            [c.metadata["chunk_id"] for c in chunks],
            vectors,
            [{"text": c.text, **c.metadata} for c in chunks],
        )
        self.vector_store.save(self.settings.vector_dir)

        # --- graph ---
        self.graph_store.ensure_constraints()
        n_entities = n_relations = 0
        for chunk in chunks:
            fragment = extract_graph_chunk(chunk.metadata["chunk_id"], chunk.text, self.llm)
            self.graph_store.upsert(
                chunk.metadata["chunk_id"],
                fragment.entities,
                fragment.relations,
                chunk_text=chunk.text,
                source=source,
                page=chunk.metadata.get("page"),
            )
            n_entities += len(fragment.entities)
            n_relations += len(fragment.relations)
        return {"chunks": len(chunks), "entities": n_entities, "relations": n_relations}


class QueryEngine:
    """Query -> hybrid retrieval (vector + graph) -> grounded, cited answer."""

    def __init__(self, embedder, vector_store, graph_store, llm, settings):
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm = llm
        self.settings = settings
        self.answerer = AnswerGenerator(llm)

    def _query_entities(self, query: str) -> List[str]:
        """Extract the entities named in the query to seed graph traversal."""
        from app.llm.prompts import EXTRACTION_SYSTEM, extraction_user_prompt

        raw = self.llm.chat(
            [
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": extraction_user_prompt(query)},
            ],
            temperature=0.0,
        )
        entities, _ = parse_extraction(raw)
        return [e.name for e in entities]

    def retrieve(self, query: str) -> List[ScoredChunk]:
        vector_hits = self.vector_store.search(
            self.embedder.embed_query(query), top_k=self.settings.top_k
        )
        graph_hits = self.graph_store.retrieve(
            self._query_entities(query),
            hops=self.settings.graph_hops,
            limit=self.settings.top_k,
        )
        return fuse(vector_hits, graph_hits, top_k=self.settings.top_k)

    @staticmethod
    def _contexts(hits: List[ScoredChunk]) -> List[Dict[str, Any]]:
        return [
            {
                "text": h.text,
                "source": h.metadata.get("source"),
                "page": h.metadata.get("page"),
                "chunk_id": h.chunk_id,
                "retriever": h.source,
                "metadata": h.metadata,
            }
            for h in hits
        ]

    def answer(self, query: str):
        contexts = self._contexts(self.retrieve(query))
        return self.answerer.generate(query, contexts), contexts

    def stream_answer(self, query: str):
        contexts = self._contexts(self.retrieve(query))
        return self.answerer.stream(query, contexts), contexts
