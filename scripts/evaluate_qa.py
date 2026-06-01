"""Phase 1C: table-only RAG QA evaluation (retrieval + LLM answer generation).

Triggered from the notebook (needs the GPU embedding stack + an OpenRouter key):
    OPENROUTER_API_KEY=... python scripts/evaluate_qa.py
    # smoke-test the key cheaply first (3 LLM calls):
    OPENROUTER_API_KEY=... python scripts/evaluate_qa.py --limit 3 --corpus gt_markdown

For each corpus ({gt,ocr} x {markdown,linearized}) this retrieves with RRF (BM25 + dense
fused - the method carried forward from the retrieval matrix), feeds the top-k chunks to the
LLM (src/llm_client, single provider, P5), and scores the answer: answer_exact /
numeric_relaxed / citation_hit (+ abstain_rate). GT and OCR are scored separately (P4) so
the OCR-vs-GT downstream answer gap is measurable. abstain_accuracy is reported once the
hand-authored unanswerable questions are in the QA set.

Writes outputs/evaluation/rag/phase1c_qa.json (metrics) and, per corpus,
outputs/qa/answers_<corpus>.jsonl (every question's retrieved ids + answer, for inspection).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.eval_qa import aggregate_qa  # noqa: E402
from src.llm_client import generate_answer  # noqa: E402
from src.retrieval import BM25Index, rrf_fuse  # noqa: E402

CORPORA = ["gt_markdown", "gt_linearized", "ocr_markdown", "ocr_linearized"]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _rrf_ranked(bm25: BM25Index, dense, question: str, top_k: int) -> list[str]:
    return rrf_fuse([bm25.search(question, top_k=top_k),
                     dense.search(question, top_k=top_k)], top_k=top_k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=10, help="chunks retrieved per question")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap questions (0 = all); use a few to smoke-test the key cheaply")
    ap.add_argument("--corpus", choices=[*CORPORA, "all"], default="all",
                    help="evaluate one corpus or all four")
    args = ap.parse_args()

    qa_path = config.QA_DIR / "qa_all.jsonl"
    if not qa_path.exists():
        raise SystemExit(f"no QA set at {qa_path} (run build_qa_dataset.py first)")
    questions = _load_jsonl(qa_path)
    if args.limit:
        questions = questions[:args.limit]
    corpora = CORPORA if args.corpus == "all" else [args.corpus]

    # One LLM call per (corpus, question): make the cost explicit before any API spend.
    print(f"QA eval: {len(corpora)} corpus x {len(questions)} questions "
          f"= {len(corpora) * len(questions)} LLM calls")

    # Colab/API pieces, loaded once and reused across corpora.
    from src.dense_retrieval import DenseIndex, build_bge_embedder
    from src.llm_client import build_openrouter_complete
    embedder = build_bge_embedder()
    complete = build_openrouter_complete()

    report = {"num_questions": len(questions), "top_k": args.top_k, "configs": {}}
    for name in corpora:
        chunk_path = config.CHUNKS / f"{name}.jsonl"
        if not chunk_path.exists():
            print(f"[skip] {name}: no corpus at {chunk_path}")
            continue
        chunks = _load_jsonl(chunk_path)
        by_id = {c["chunk_id"]: c for c in chunks}
        bm25 = BM25Index(chunks)
        dense = DenseIndex(chunks, embedder)

        results, records = [], []
        for q in questions:
            ranked = _rrf_ranked(bm25, dense, q["question"], args.top_k)
            evidence = [by_id[cid] for cid in ranked if cid in by_id]
            ans = generate_answer(q["question"], evidence, complete=complete)
            results.append({
                "pred": ans.answer, "gold": q.get("gold_answer", ""),
                "citations": ans.citations, "relevant": q.get("relevant_chunk_ids", []),
                "abstained": ans.abstained, "is_answerable": q.get("is_answerable", True),
            })
            records.append({
                "question_id": q.get("question_id"), "question": q["question"],
                "gold_answer": q.get("gold_answer", ""), "answer": ans.answer,
                "citations": ans.citations, "abstained": ans.abstained,
                "retrieved": ranked, "relevant_chunk_ids": q.get("relevant_chunk_ids", []),
            })

        report["configs"][name] = aggregate_qa(results)
        answers_path = config.QA_DIR / f"answers_{name}.jsonl"
        answers_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8")

    out_dir = config.EVALUATION / "rag"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1c_qa.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary: one row per config.
    cols = ["answer_exact", "numeric_relaxed", "citation_hit", "abstain_rate"]
    print(f"{'config':<16} " + " ".join(f"{c:>15}" for c in cols))
    for name, m in report["configs"].items():
        if "answer_exact" not in m:
            continue
        cells = []
        for c in cols:
            v = m.get(c)
            cells.append("    n/a" if v is None else f"{v:>15.3f}")
        print(f"{name:<16} " + " ".join(cells) + f"   (n={m['num_answerable']})")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
