"""Phase 1B content error analysis: per-cell GT vs OCR text diff (CPU).

Pairs ocr_filled with gt_filled for topology-matched samples (num_rows and num_cols equal,
so cells line up 1-to-1 by grid index) and classifies every cell, printing a tally per
sample plus a few example diffs. This localises whether low content scores come from OCR
misreads, lost text, spurious text, or cells filled with the wrong words. It only reads
tables and prints - it changes no metric.

Cell classes:
  EXACT      GT and OCR text identical (after whitespace collapse)
  NUM_OK     numeric GT cell, not exact but matches under relaxed numeric comparison
  NUM_DIFF   numeric GT cell, value does not match
  MISS       GT has text, OCR cell is empty
  SPURIOUS   OCR has text, GT cell is empty
  TEXT_DIFF  both non-empty, non-numeric, and different

    python scripts/diff_content.py --run-id debug_nomkl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.numeric_utils import looks_numeric, relaxed_numeric_match  # noqa: E402
from src.run_manifest import read_completed  # noqa: E402

OCR_PHASE = "phase1b_ocr"


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def _classify(gt_text: str, ocr_text: str) -> str:
    if gt_text == ocr_text:
        return "EXACT"
    if looks_numeric(gt_text):
        return "NUM_OK" if relaxed_numeric_match(ocr_text, gt_text) else "NUM_DIFF"
    if gt_text and not ocr_text:
        return "MISS"
    if ocr_text and not gt_text:
        return "SPURIOUS"
    return "TEXT_DIFF"


def _anchor_text(table: dict) -> dict:
    return {(c["row_start"], c["col_start"]): _norm(c.get("text", ""))
            for c in table["cells"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="debug", help="the ocr_filled run-id")
    ap.add_argument("--max-samples", type=int, default=3,
                    help="how many topology-matched samples to show")
    ap.add_argument("--max-diffs", type=int, default=30,
                    help="max example diffs printed per sample")
    ap.add_argument("--trunc", type=int, default=120, help="truncate cell text to N chars")
    args = ap.parse_args()

    rows = read_completed(config.MANIFESTS / f"{OCR_PHASE}_{args.run_id}.csv")
    if not rows:
        raise SystemExit(f"no completed samples for run-id {args.run_id}")

    shown = 0
    for row in rows:
        sample_id = row["sample_id"]
        pred_path = Path(row["output_path"])
        gt_path = config.TABLES_GT_FILLED / f"{sample_id}.json"
        if not pred_path.exists() or not gt_path.exists():
            continue

        ocr = json.loads(pred_path.read_text(encoding="utf-8"))
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        # Only topology-matched samples, so (row,col) indices align 1-to-1 and the diff
        # reflects text, not topology.
        if ocr["num_rows"] != gt["num_rows"] or ocr["num_cols"] != gt["num_cols"]:
            continue

        g = _anchor_text(gt)
        o = _anchor_text(ocr)
        tally: Counter = Counter()
        diffs_shown = 0

        print("=" * 100)
        print(f"SAMPLE: {sample_id}  {ocr['num_rows']}x{ocr['num_cols']}")
        for k in sorted(g):
            gg, oo = g[k], o.get(k, "")
            kind = _classify(gg, oo)
            tally[kind] += 1
            if kind != "EXACT" and diffs_shown < args.max_diffs:
                print(f"{k} [{kind}]")
                print(f"  GT : {gg[:args.trunc]!r}")
                print(f"  OCR: {oo[:args.trunc]!r}")
                diffs_shown += 1
        print(f"tally: {dict(tally)}")

        shown += 1
        if shown >= args.max_samples:
            break

    if shown == 0:
        print("no topology-matched samples found for this run-id")


if __name__ == "__main__":
    main()
