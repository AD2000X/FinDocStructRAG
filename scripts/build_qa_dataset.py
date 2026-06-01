"""Phase 1C (CPU): build the QA evaluation set.

Triggered from the notebook:
    !python scripts/build_qa_dataset.py --limit 30 --seed 42

Generates templated lookup/numeric QA from the GT-filled tables (answer = GT cell, so
gold answers and relevance judgments are automatic), samples them across tables for issuer
spread, then merges the hand-authored manual + unanswerable seed (config.QA_MANUAL_SEED)
if present. Writes:

    outputs/qa/templated.jsonl   the sampled templated questions
    outputs/qa/qa_all.jsonl      templated + manual seed (the eval set)

Gold answers come from GT only; the eval (evaluate_rag.py) runs the pipeline over the
GT-filled and OCR-filled corpora separately (P4) to measure the OCR-vs-GT downstream gap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.qa_templates import generate_lookup_questions  # noqa: E402


def _sample_across_tables(per_table: list[list[dict]], limit: int, seed: int) -> list[dict]:
    """Round-robin one question at a time from each (shuffled) table until limit is met.

    Round-robin (not a flat shuffle) so the set spreads across tables/issuers instead of
    concentrating on whichever tables have the most cells.
    """
    rng = random.Random(seed)
    queues = [lst[:] for lst in per_table if lst]
    for q in queues:
        rng.shuffle(q)
    rng.shuffle(queues)

    out: list[dict] = []
    while queues and len(out) < limit:
        for q in list(queues):
            if len(out) >= limit:
                break
            out.append(q.pop())
            if not q:
                queues.remove(q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30, help="number of templated questions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gt_dir = config.TABLES_GT_FILLED
    if not gt_dir.exists():
        raise SystemExit(f"no gt_filled tables at {gt_dir}")

    per_table = []
    for table_path in sorted(gt_dir.glob("*.json")):
        table = json.loads(table_path.read_text(encoding="utf-8"))
        records = generate_lookup_questions(table)
        if records:
            per_table.append(records)

    templated = _sample_across_tables(per_table, args.limit, args.seed)
    for i, rec in enumerate(templated, start=1):
        rec["question_id"] = f"tq_{i:04d}"

    config.QA_DIR.mkdir(parents=True, exist_ok=True)
    templated_path = config.QA_DIR / "templated.jsonl"
    templated_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in templated),
        encoding="utf-8")

    manual = []
    if config.QA_MANUAL_SEED.exists():
        manual = [json.loads(line) for line in
                  config.QA_MANUAL_SEED.read_text(encoding="utf-8").splitlines()
                  if line.strip()]

    all_path = config.QA_DIR / "qa_all.jsonl"
    all_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [*templated, *manual]),
        encoding="utf-8")

    n_unans = sum(1 for r in manual if not r.get("is_answerable", True))
    if manual:
        seed_msg = f"manual seed    : {len(manual)} ({n_unans} unanswerable)"
    elif config.QA_MANUAL_SEED.exists():
        seed_msg = f"manual seed    : 0 ({config.QA_MANUAL_SEED} is empty - author it)"
    else:
        seed_msg = f"manual seed    : 0 ({config.QA_MANUAL_SEED} not found - author it)"
    print(f"templated      : {len(templated)} (from {len(per_table)} tables with questions)")
    print(seed_msg)
    print(f"qa_all total   : {len(templated) + len(manual)}")
    print(f"templated -> {templated_path}")
    print(f"qa_all    -> {all_path}")


if __name__ == "__main__":
    main()
