"""Phase 1C: table-only RAG retrieval evaluation (BM25 / dense / RRF).

Triggered from the notebook:
    !python scripts/evaluate_rag.py                       # BM25 only (CPU, no key)
    !python scripts/evaluate_rag.py --methods bm25,dense,rrf   # + dense (Colab GPU)

Runs the same QA set (outputs/qa/qa_all.jsonl) against each retrieval corpus -
{gt,ocr} x {markdown,linearized} - and reports hit@k / recall@k / MRR@k per method. GT and
OCR are scored separately (P4) so the markdown-vs-linearized serialization and the
OCR-vs-GT degradation are both directly comparable. No LLM, no API key (P5); answer
generation is a later step.

BM25 is pure CPU. dense (BGE embeddings) and rrf (RRF of BM25 + dense) need the Colab
embedding stack, so they are opt-in via --methods; the default stays the CPU/no-key path.
Writes outputs/evaluation/rag/phase1c_retrieval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.eval_retrieval import aggregate_retrieval  # noqa: E402
from src.retrieval import BM25Index, rrf_fuse  # noqa: E402

KS = (1, 5, 10)
CORPORA = ["gt_markdown", "gt_linearized", "ocr_markdown", "ocr_linearized"]
METHODS = ("bm25", "dense", "rrf")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _rankings(index, questions: list[dict], top_k: int) -> list[list[str]]:
    return [index.search(q["question"], top_k=top_k) for q in questions]


def _aggregate(ranked_lists: list[list[str]], relevants: list[list[str]]) -> dict:
    results = [{"ranked": r, "relevant": rel}
               for r, rel in zip(ranked_lists, relevants)]
    return aggregate_retrieval(results, ks=KS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=max(KS))
    ap.add_argument("--methods", default="bm25",
                    help=f"comma list of {','.join(METHODS)} (dense/rrf need the GPU stack)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"unknown method(s): {bad}; choose from {list(METHODS)}")
    need_dense = any(m in ("dense", "rrf") for m in methods)

    qa_path = config.QA_DIR / "qa_all.jsonl"
    if not qa_path.exists():
        raise SystemExit(f"no QA set at {qa_path} (run build_qa_dataset.py first)")
    questions = _load_jsonl(qa_path)
    relevants = [q.get("relevant_chunk_ids", []) for q in questions]

    # The BGE model (Colab GPU) is loaded once and reused across corpora.
    embedder = None
    if need_dense:
        from src.dense_retrieval import DenseIndex, build_bge_embedder
        embedder = build_bge_embedder()

    report = {"num_questions": len(questions), "ks": list(KS),
              "methods": methods, "corpora": {}}
    for name in CORPORA:
        chunk_path = config.CHUNKS / f"{name}.jsonl"
        if not chunk_path.exists():
            print(f"[skip] {name}: no corpus at {chunk_path}")
            continue
        chunks = _load_jsonl(chunk_path)

        bm25_ranked = _rankings(BM25Index(chunks), questions, args.top_k)
        dense_ranked = None
        if need_dense:
            dense_ranked = _rankings(DenseIndex(chunks, embedder), questions, args.top_k)

        per_method = {}
        for m in methods:
            if m == "bm25":
                ranked_lists = bm25_ranked
            elif m == "dense":
                ranked_lists = dense_ranked
            else:  # rrf: fuse BM25 + dense per question
                ranked_lists = [rrf_fuse([b, d], top_k=args.top_k)
                                for b, d in zip(bm25_ranked, dense_ranked)]
            per_method[m] = _aggregate(ranked_lists, relevants)
        report["corpora"][name] = per_method

    out_dir = config.EVALUATION / "rag"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1c_retrieval.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary: one row per (corpus, method).
    cols = [f"hit@{k}" for k in KS] + [f"mrr@{k}" for k in KS]
    print(f"{'corpus':<16} {'method':<6} " + " ".join(f"{c:>8}" for c in cols))
    for name, methods_dict in report["corpora"].items():
        for m, metrics in methods_dict.items():
            if "hit@1" not in metrics:
                continue
            print(f"{name:<16} {m:<6} "
                  + " ".join(f"{metrics[c]:>8.3f}" for c in cols)
                  + f"   (n={metrics['num_answerable']})")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
