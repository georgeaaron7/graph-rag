"""llm-based knowledge-graph extraction from text chunks.

each chunk is sent to the llm with a strict json-schema request; we parse the
result into typed :class:`Entity` / :class:`Relation` objects and stamp them
with the originating ``chunk_id`` so every graph edge can be traced back to the
exact text span that produced it. That provenance is what powers citations.

the json parser is deliberately defensive: models wrap json in prose or code
fences, so one malformed chunk must never break a whole ingestion run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple


@dataclass
class Entity:
    name: str
    type: str = "Entity"


@dataclass
class Relation:
    source: str
    target: str
    type: str = "RELATED_TO"


@dataclass
class GraphChunk:
    chunk_id: str
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return raw


def parse_extraction(raw: str) -> Tuple[List[Entity], List[Relation]]:
    """Parse the LLM's JSON response into entities + relations.

    Tolerates code fences, prose around the JSON, and missing keys. Returns
    empty lists rather than raising, and de-duplicates entities case-insensitively.
    """
    text = _strip_code_fences(raw or "")
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return [], []
    if not isinstance(data, dict):
        return [], []

    entities: List[Entity] = []
    seen = set()
    for item in data.get("entities", []) or []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "Entity")).strip() or "Entity"
        else:
            name, etype = str(item).strip(), "Entity"
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            entities.append(Entity(name=name, type=etype))

    relations: List[Relation] = []
    raw_rels = data.get("relationships") or data.get("relations") or []
    for item in raw_rels:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "")).strip()
        tgt = str(item.get("target", "")).strip()
        rtype = str(item.get("type", "RELATED_TO")).strip() or "RELATED_TO"
        if src and tgt:
            relations.append(Relation(source=src, target=tgt, type=rtype))
    return entities, relations


def extract_graph_chunk(chunk_id: str, text: str, llm: Any) -> GraphChunk:
    """Call the LLM to extract one chunk's graph fragment (entities + relations)."""
    from app.llm.prompts import EXTRACTION_SYSTEM, extraction_user_prompt

    raw = llm.chat(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": extraction_user_prompt(text)},
        ],
        temperature=0.0,
    )
    entities, relations = parse_extraction(raw)
    return GraphChunk(chunk_id=chunk_id, entities=entities, relations=relations)
