"""Compare vector-only vs hybrid (graph+vector) retrieval on a labelled QA set.

Primary metrics are computed here with no external dependencies:
  * Hit@k  - did any gold chunk appear in the top-k?
  * MRR    - reciprocal rank of the first gold chunk.

This is the core evidence for "GraphRAG beats plain RAG": run it and put the
before/after numbers in your README. Build the QA set by ingesting a document,
finding the chunk_ids that truly answer each question (via the Neo4j browser or
the /chat citations), and listing them as gold_chunk_ids in sample_qa.json.

Optional: if `ragas` is installed you can extend this to score answer
faithfulness / relevancy - see the note at the bottom.

Usage:
    python eval/evaluate.py                 # uses eval/sample_qa.json
    python eval/evaluate.py my_qa.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.config import get_settings  # noqa: E402
from app.graph.store import GraphStore  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.pipeline import QueryEngine  # noqa: E402
from app.vector.embeddings import Embedder  # noqa: E402
from app.vector.store import VectorStore  # noqa: E402


def hit_at_k(retrieved_ids, gold_ids, k):
    return 1.0 if set(retrieved_ids[:k]) & set(gold_ids) else 0.0


def mrr(retrieved_ids, gold_ids):
    for i, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_ids:
            return 1.0 / i
    return 0.0


def main() -> None:
    qa_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "sample_qa.json"
    )
    with open(qa_path) as f:
        qa = [item for item in json.load(f) if not str(item.get("question", "")).startswith("_")]

    settings = get_settings()
    llm = LLMClient(settings.llm_model, settings.llm_base_url, settings.llm_api_key)
    embedder = Embedder(
        settings.embedding_backend,
        settings.embedding_model,
        settings.llm_base_url,
        settings.llm_api_key,
    )
    graph = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    vector_store = VectorStore.load(settings.vector_dir)
    engine = QueryEngine(embedder, vector_store, graph, llm, settings)

    rows = []
    for item in qa:
        question = item["question"]
        gold = item.get("gold_chunk_ids", [])

        vector_ids = [h.chunk_id for h in vector_store.search(embedder.embed_query(question), top_k=settings.top_k)]
        hybrid_ids = [h.chunk_id for h in engine.retrieve(question)]

        rows.append(
            {
                "vector_hit@5": hit_at_k(vector_ids, gold, 5),
                "vector_mrr": mrr(vector_ids, gold),
                "hybrid_hit@5": hit_at_k(hybrid_ids, gold, 5),
                "hybrid_mrr": mrr(hybrid_ids, gold),
            }
        )

    def avg(key):
        return round(sum(r[key] for r in rows) / max(1, len(rows)), 3)

    print(
        json.dumps(
            {
                "n_questions": len(rows),
                "vector_only": {"hit@5": avg("vector_hit@5"), "mrr": avg("vector_mrr")},
                "hybrid_graphrag": {"hit@5": avg("hybrid_hit@5"), "mrr": avg("hybrid_mrr")},
            },
            indent=2,
        )
    )


# --- Optional RAGAS extension -------------------------------------------------
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# Build a datasets.Dataset with columns: question, answer, contexts, ground_truth
# then: evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
# Report the scores alongside the retrieval metrics above.

if __name__ == "__main__":
    main()
