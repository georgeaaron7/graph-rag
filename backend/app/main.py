"""FastAPI entrypoint for the GraphRAG Knowledge Assistant.

Run from the ``backend/`` directory:

    uvicorn app.main:app --reload

Endpoints: ``GET /health``, ``POST /ingest`` (PDF upload), ``POST /chat``
(answer + citations), ``POST /chat/stream`` (SSE token stream + citations).
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, Citation, IngestResponse

app = FastAPI(title="GraphRAG Knowledge Assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-process singletons. For multi-worker deployments, move these behind
# a shared service (the stores already persist to disk / Neo4j).
_state: Dict[str, object] = {}


def _init() -> None:
    if _state:
        return
    from app.graph.store import GraphStore
    from app.llm.client import LLMClient
    from app.pipeline import IngestionPipeline
    from app.vector.embeddings import Embedder
    from app.vector.store import VectorStore

    settings = get_settings()
    llm = LLMClient(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    embedder = Embedder(
        backend=settings.embedding_backend,
        model=settings.embedding_model,
        api_base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )
    graph = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    vector_store = None
    if os.path.exists(os.path.join(settings.vector_dir, "index.faiss")):
        vector_store = VectorStore.load(settings.vector_dir)
    _state.update(
        settings=settings,
        llm=llm,
        embedder=embedder,
        graph=graph,
        vector_store=vector_store,
        ingest=IngestionPipeline(embedder, VectorStore, graph, llm, settings),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    _init()
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        stats = _state["ingest"].ingest_pdf(tmp_path)  # type: ignore[attr-defined]
        _state["vector_store"] = _state["ingest"].vector_store  # type: ignore[attr-defined]
    finally:
        os.unlink(tmp_path)
    return IngestResponse(source=file.filename, **stats)


def _query_engine():
    from app.pipeline import QueryEngine

    if _state.get("vector_store") is None:
        raise HTTPException(status_code=400, detail="No documents ingested. POST /ingest first.")
    return QueryEngine(
        _state["embedder"], _state["vector_store"], _state["graph"], _state["llm"], _state["settings"]
    )


def _citations(contexts) -> List[Citation]:
    return [
        Citation(
            index=i,
            chunk_id=c["chunk_id"],
            source=c.get("source"),
            page=c.get("page"),
            retriever=c.get("retriever", "hybrid"),
            snippet=(c.get("text") or ""),
        )
        for i, c in enumerate(contexts, start=1)
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    _init()
    answer, contexts = _query_engine().answer(req.query)
    return ChatResponse(answer=answer, citations=_citations(contexts))


@app.post("/retrieve")
def retrieve(req: ChatRequest):
    """Retrieval-only endpoint (no answer generation).

    Returns the ranked hybrid chunks so an external caller — e.g. the AgentFlow
    research assistant using this as its ``doc_search`` tool — can reason over the
    evidence itself.
    """
    _init()
    engine = _query_engine()
    contexts = engine._contexts(engine.retrieve(req.query))
    return {"query": req.query, "contexts": contexts}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    _init()
    token_iter, contexts = _query_engine().stream_answer(req.query)

    def event_stream():
        citations = [c.model_dump() for c in _citations(contexts)]
        yield f"event: citations\ndata: {json.dumps(citations)}\n\n"
        for token in token_iter:
            yield f"event: token\ndata: {json.dumps({'t': token})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
