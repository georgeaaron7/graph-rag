"""api request/response models."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    source: str
    chunks: int
    entities: int
    relations: int


class Citation(BaseModel):
    index: int
    chunk_id: str
    source: Optional[str] = None
    page: Optional[int] = None
    retriever: str = "hybrid"  # vector | graph | hybrid
    snippet: str = ""


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
