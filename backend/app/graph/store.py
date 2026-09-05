"""Neo4j knowledge-graph store: ingestion + multi-hop retrieval.

Graph model
-----------
``(:Entity {name, type})`` nodes, connected to each other by ``[:REL {type}]``
edges (the extracted relationships) and to ``(:Chunk {id, text, source, page})``
nodes by ``[:MENTIONED_IN]``. Retrieval starts from the entities named in the
query, walks up to *N* hops across ``:REL`` edges, and returns the chunks those
connected entities are mentioned in - the multi-hop step plain vector search
cannot do.
"""
from __future__ import annotations

from typing import Any, List, Sequence

from app.graph.extraction import Entity, Relation
from app.retrieval.hybrid import ScoredChunk


class GraphStore:
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def ensure_constraints(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
            )

    def upsert(
        self,
        chunk_id: str,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        chunk_text: str = "",
        source: str = "",
        page: Any = None,
    ) -> None:
        with self._driver.session() as session:
            session.execute_write(
                self._upsert_tx, chunk_id, entities, relations, chunk_text, source, page
            )

    @staticmethod
    def _upsert_tx(tx, chunk_id, entities, relations, chunk_text, source, page):
        tx.run(
            "MERGE (c:Chunk {id:$id}) SET c.text=$text, c.source=$source, c.page=$page",
            id=chunk_id, text=chunk_text, source=source, page=page,
        )
        for entity in entities:
            tx.run(
                "MERGE (e:Entity {name:$name}) SET e.type=coalesce(e.type,$type) "
                "WITH e MATCH (c:Chunk {id:$cid}) MERGE (e)-[:MENTIONED_IN]->(c)",
                name=entity.name, type=entity.type, cid=chunk_id,
            )
        for relation in relations:
            tx.run(
                "MERGE (a:Entity {name:$src}) MERGE (b:Entity {name:$tgt}) "
                "MERGE (a)-[rel:REL {type:$type}]->(b)",
                src=relation.source, tgt=relation.target, type=relation.type,
            )

    def retrieve(
        self, entity_names: Sequence[str], hops: int = 2, limit: int = 8
    ) -> List[ScoredChunk]:
        """From seed entities, walk up to ``hops`` ``:REL`` edges and return the
        chunks the connected entities are mentioned in."""
        if not entity_names:
            return []
        # ``hops`` is cast to int and interpolated (Cypher can't parametrize a
        # variable-length path bound); everything else is parametrized.
        cypher = (
            "UNWIND $names AS nm "
            "MATCH (seed:Entity) WHERE toLower(seed.name) CONTAINS toLower(nm) "
            f"MATCH (seed)-[:REL*0..{int(hops)}]-(connected:Entity) "
            "MATCH (connected)-[:MENTIONED_IN]->(c:Chunk) "
            "RETURN DISTINCT c.id AS chunk_id, c.text AS text, "
            "c.source AS source, c.page AS page LIMIT $limit"
        )
        hits: List[ScoredChunk] = []
        with self._driver.session() as session:
            for i, rec in enumerate(session.run(cypher, names=list(entity_names), limit=limit)):
                hits.append(
                    ScoredChunk(
                        chunk_id=rec["chunk_id"],
                        text=rec.get("text") or "",
                        score=1.0 / (i + 1),  # positional; real ranking happens in fuse()
                        source="graph",
                        metadata={"source": rec.get("source"), "page": rec.get("page")},
                    )
                )
        return hits
