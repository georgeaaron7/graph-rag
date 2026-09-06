"""CLI: ingest a PDF/text file (or a folder of them) into the vector + graph stores.

Usage (from the project root, with backend deps installed and Neo4j running):

    python scripts/ingest.py data/sample/mydoc.pdf
    python scripts/ingest.py data/sample/            # all PDFs + .txt/.md in a folder
    python scripts/ingest.py data/multihop/          # e.g. a benchmark text corpus
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.config import get_settings  # noqa: E402
from app.graph.store import GraphStore  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.pipeline import IngestionPipeline  # noqa: E402
from app.vector.embeddings import Embedder  # noqa: E402
from app.vector.store import VectorStore  # noqa: E402

PDF_EXT = (".pdf",)
TEXT_EXT = (".txt", ".md")


def _collect(target: str) -> list:
    if os.path.isdir(target):
        paths = []
        for ext in PDF_EXT + TEXT_EXT:
            paths.extend(glob.glob(os.path.join(target, f"*{ext}")))
        return sorted(paths)
    return [target] if target.lower().endswith(PDF_EXT + TEXT_EXT) else []


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/ingest.py <file_or_folder>  (.pdf, .txt, .md)")
        sys.exit(1)

    target = sys.argv[1]
    paths = _collect(target)
    if not paths:
        print(f"No .pdf/.txt/.md files found at {target!r}.")
        sys.exit(1)

    settings = get_settings()
    llm = LLMClient(settings.llm_model, settings.llm_base_url, settings.llm_api_key)
    embedder = Embedder(
        settings.embedding_backend,
        settings.embedding_model,
        settings.llm_base_url,
        settings.llm_api_key,
    )
    graph = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    pipeline = IngestionPipeline(embedder, VectorStore, graph, llm, settings)

    totals = {"chunks": 0, "entities": 0, "relations": 0}
    for path in paths:
        print(f"Ingesting {path} ...")
        if path.lower().endswith(PDF_EXT):
            stats = pipeline.ingest_pdf(path)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            stats = pipeline.ingest_text(text, source=os.path.basename(path))
        for key in totals:
            totals[key] += stats[key]
        print(f"  -> {stats}")

    print(f"\nDone. Totals: {totals}")
    print(f"Vector index saved to {settings.vector_dir}")
    print("Explore the graph at http://localhost:7474")


if __name__ == "__main__":
    main()
