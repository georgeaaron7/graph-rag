#!/usr/bin/env python3
"""Wire the MultiHop-RAG benchmark into this repo's retrieval eval.

MultiHop-RAG (yixuantt/MultiHopRAG on the Hugging Face Hub, ODC-BY) is a
purpose-built multi-hop RAG benchmark: ~2,556 queries whose supporting evidence
is spread across several news articles, plus the article corpus. Multi-hop is
exactly where hybrid graph+vector retrieval should beat vector-only, so it's the
strongest public dataset to validate the Hit@k / MRR gap this project claims.

This runs in two phases because the repo scores retrieval against *in-repo*
chunk ids (``{source}::p{page}::c{index}``), which only exist after ingestion:

  1) prepare  — download the corpus + queries and write:
                  data/multihop/*.txt        (one file per article -> the corpus)
                  eval/multihop_queries.json  (query + provided gold evidence)

  2) (you)    — ingest the corpus so chunks + the FAISS sidecar exist:
                  python scripts/ingest.py data/multihop/

  3) build-qa — read the saved vector store (data/index/store.json), match each
                query's provided evidence text to the chunk ids that contain it,
                and write eval/sample_qa.json in the format evaluate.py expects.

Then:
    python eval/evaluate.py     # Hit@5 / MRR, vector-only vs hybrid

NOTE on honesty: build-qa derives gold_chunk_ids automatically by matching the
benchmark's own evidence sentences to your chunks (distant supervision). Spot-check
a handful in eval/sample_qa.json before quoting the numbers.

Field names on the HF dataset can shift between versions; this script prints the
columns it sees and falls back across common names. Verify against `ds.features`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
CORPUS_DIR = os.path.join(ROOT, "data", "multihop")
QUERIES_JSON = os.path.join(ROOT, "eval", "multihop_queries.json")
STORE_JSON = os.path.join(ROOT, "data", "index", "store.json")
SAMPLE_QA = os.path.join(ROOT, "eval", "sample_qa.json")

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list:
    return _WORD.findall(text.lower())


def _first(d: dict, keys, default=""):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return default


# ---------------------------------------------------------------------------
# Pure matcher (unit-tested): evidence sentence -> chunk_ids that contain it.
# ---------------------------------------------------------------------------
def containment(evidence: str, chunk_text: str) -> float:
    """Fraction of the evidence's tokens present in the chunk (0..1)."""
    ev = _tokens(evidence)
    if not ev:
        return 0.0
    chunk = set(_tokens(chunk_text))
    hit = sum(1 for t in ev if t in chunk)
    return hit / len(ev)


def match_evidence_to_chunks(evidences, chunks, threshold=0.6, per_evidence=2):
    """Return gold chunk_ids for one query.

    ``chunks`` is a dict {chunk_id: text}. For each evidence string, keep the
    top ``per_evidence`` chunks whose token-containment >= ``threshold``.
    """
    gold = []
    for ev in evidences:
        scored = (
            (cid, containment(ev, text)) for cid, text in chunks.items()
        )
        ranked = sorted(
            (cs for cs in scored if cs[1] >= threshold),
            key=lambda cs: cs[1],
            reverse=True,
        )
        for cid, _ in ranked[:per_evidence]:
            if cid not in gold:
                gold.append(cid)
    return gold


# ---------------------------------------------------------------------------
# Phase 1: download corpus + queries from HF.
# ---------------------------------------------------------------------------
def cmd_prepare(args) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets  (needed only for `prepare`)")

    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(QUERIES_JSON), exist_ok=True)

    # --- corpus ---
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    print(f"corpus columns: {corpus.column_names}")
    n = 0
    for i, row in enumerate(corpus):
        body = _first(row, ["body", "text", "passage", "content", "document"])
        title = _first(row, ["title", "source", "id"], default=f"doc{i}")
        if not body:
            continue
        safe = re.sub(r"[^A-Za-z0-9]+", "_", str(title))[:60] or f"doc{i}"
        with open(os.path.join(CORPUS_DIR, f"{safe}_{i}.txt"), "w", encoding="utf-8") as fh:
            fh.write(body)
        n += 1
        if args.max_docs and n >= args.max_docs:
            break
    print(f"wrote {n} corpus files -> {CORPUS_DIR}")

    # --- queries ---
    qa = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    print(f"query columns: {qa.column_names}")
    out = []
    for i, row in enumerate(qa):
        query = _first(row, ["query", "question"])
        answer = _first(row, ["answer", "label"])
        ev_raw = _first(row, ["evidence_list", "evidences", "evidence", "supporting_facts"], default=[])
        evidences = []
        if isinstance(ev_raw, list):
            for e in ev_raw:
                if isinstance(e, dict):
                    evidences.append(_first(e, ["fact", "text", "evidence", "sentence", "title"]))
                elif isinstance(e, str):
                    evidences.append(e)
        qtype = _first(row, ["question_type", "type"], default="multi_hop")
        if query and evidences:
            out.append({"question": query, "answer": answer, "evidences": [e for e in evidences if e], "type": qtype})
        if args.max_queries and len(out) >= args.max_queries:
            break
    with open(QUERIES_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {len(out)} queries -> {QUERIES_JSON}")
    print("\nNext: python scripts/ingest.py data/multihop/   then   "
          "python scripts/prepare_multihoprag.py build-qa")


# ---------------------------------------------------------------------------
# Phase 3: build sample_qa.json from ingested chunks + provided evidence.
# ---------------------------------------------------------------------------
def cmd_build_qa(args) -> None:
    if not os.path.exists(STORE_JSON):
        sys.exit(f"{STORE_JSON} not found — ingest the corpus first: python scripts/ingest.py data/multihop/")
    if not os.path.exists(QUERIES_JSON):
        sys.exit(f"{QUERIES_JSON} not found — run `prepare` first.")

    with open(STORE_JSON, encoding="utf-8") as fh:
        store = json.load(fh)
    chunks = {cid: rec.get("text", "") for cid, rec in store.get("meta", {}).items()}
    print(f"{len(chunks)} chunks in the vector store")

    with open(QUERIES_JSON, encoding="utf-8") as fh:
        queries = json.load(fh)

    qa, dropped = [], 0
    for q in queries:
        gold = match_evidence_to_chunks(
            q.get("evidences", []), chunks, threshold=args.threshold, per_evidence=args.per_evidence
        )
        if not gold:
            dropped += 1
            continue  # no chunk matched this query's evidence — can't score it fairly
        qa.append({"question": q["question"], "gold_chunk_ids": gold, "type": q.get("type", "multi_hop")})
        if args.max_queries and len(qa) >= args.max_queries:
            break

    with open(SAMPLE_QA, "w", encoding="utf-8") as fh:
        json.dump(qa, fh, indent=2)
    print(f"wrote {len(qa)} QA items -> {SAMPLE_QA}  (dropped {dropped} with no evidence match)")
    print("Spot-check a few gold_chunk_ids, then: python eval/evaluate.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download corpus + queries from HF")
    p.add_argument("--max-docs", type=int, default=0, help="cap corpus files (0 = all; AuraDB Free has a node cap)")
    p.add_argument("--max-queries", type=int, default=0, help="cap queries (0 = all)")
    p.set_defaults(func=cmd_prepare)

    b = sub.add_parser("build-qa", help="derive eval/sample_qa.json from ingested chunks")
    b.add_argument("--threshold", type=float, default=0.6, help="min token-containment to treat a chunk as gold")
    b.add_argument("--per-evidence", type=int, default=2, help="max chunks kept per evidence sentence")
    b.add_argument("--max-queries", type=int, default=0, help="cap QA items written (0 = all)")
    b.set_defaults(func=cmd_build_qa)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
