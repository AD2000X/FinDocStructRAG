"""Phase 1C (CPU): table-only RAG retrieval evaluation.

Triggered from the notebook:
    !python scripts/evaluate_rag.py

Runs the same QA set (outputs/qa/qa_all.jsonl) against each retrieval corpus -
{gt,ocr} x {markdown,linearized} - with BM25, and reports hit@k / recall@k / MRR@k. GT and
OCR are scored separately (P4) so the markdown-vs-linearized serialization and the
OCR-vs-GT degradation are both directly comparable. No LLM, no API key (P5); answer
generation is a later step. Writes outputs/evaluation/rag/phase1c_retrieval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.eval_retrieval import aggregate_retrieval  # noqa: E402
from src.retrieval import BM25Index  # noqa: E402

KS = (1, 5, 10)
CORPORA = ["gt_markdown", "gt_linearized", "ocr_markdown", "ocr_linearized"]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _evaluate_corpus(chunks: list[dict], questions: list[dict], top_k: int) -> dict:
    index = BM25Index(chunks)
    results = [
        {"ranked": index.search(q["question"], top_k=top_k),
         "relevant": q.get("relevant_chunk_ids", [])}
        for q in questions
    ]
    return aggregate_retrieval(results, ks=KS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=max(KS))
    args = ap.parse_args()

    qa_path = config.QA_DIR / "qa_all.jsonl"
    if not qa_path.exists():
        raise SystemExit(f"no QA set at {qa_path} (run build_qa_dataset.py first)")
    questions = _load_jsonl(qa_path)

    report = {"num_questions": len(questions), "ks": list(KS), "corpora": {}}
    for name in CORPORA:
        chunk_path = config.CHUNKS / f"{name}.jsonl"
        if not chunk_path.exists():
            print(f"[skip] {name}: no corpus at {chunk_path}")
            continue
        chunks = _load_jsonl(chunk_path)
        report["corpora"][name] = _evaluate_corpus(chunks, questions, args.top_k)

    out_dir = config.EVALUATION / "rag"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1c_retrieval.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary: one row per corpus.
    cols = [f"hit@{k}" for k in KS] + [f"mrr@{k}" for k in KS]
    print(f"{'corpus':<16} " + " ".join(f"{c:>8}" for c in cols))
    for name, m in report["corpora"].items():
        if "hit@1" not in m:
            continue
        print(f"{name:<16} " + " ".join(f"{m[c]:>8.3f}" for c in cols)
              + f"   (n={m['num_answerable']})")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
