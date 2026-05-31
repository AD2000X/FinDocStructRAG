"""Recompute Phase 1B content metrics from persisted ocr_filled vs gt_filled (CPU).

Reads the ocr_filled run manifest, takes every sample whose final status is success,
loads its ocr_filled table (the manifest's output_path) and the matching gt_filled table
(outputs/tables/gt_filled/<sample_id>.json), and aggregates content metrics over all
pairs. Like evaluate_tables.py, scoring runs over every persisted pair regardless of how
many sessions produced them.

    python scripts/evaluate_content.py --run-id debug

Requires gt_filled/ (run_phase1b_gt_filled.py) and ocr_filled/ (run_phase1b_ocr_filled.py)
to exist for the subset. Safe to re-run; only reads tables and rewrites the report. The
metrics it calls are unit-tested in tests/ (P3); this script is the glue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.eval_content import (  # noqa: E402
    aggregate_content,
    content_sample_counts,
    write_content_report,
)
from src.run_manifest import read_completed  # noqa: E402

OCR_PHASE = "phase1b_ocr"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="debug", help="the ocr_filled run-id")
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="min bbox IoU for a GT cell to match a pred cell")
    args = ap.parse_args()

    manifest_path = config.MANIFESTS / f"{OCR_PHASE}_{args.run_id}.csv"
    rows = read_completed(manifest_path)
    if not rows:
        raise SystemExit(f"no completed samples in {manifest_path}")

    per_sample: list[dict] = []
    missing_pred = missing_gt = 0
    for row in rows:
        sample_id = row["sample_id"]
        pred_path = Path(row["output_path"])
        gt_path = config.TABLES_GT_FILLED / f"{sample_id}.json"
        if not pred_path.exists():
            missing_pred += 1
            continue
        if not gt_path.exists():
            missing_gt += 1
            continue
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        per_sample.append(
            content_sample_counts(pred, gt, iou_threshold=args.iou_threshold))

    summary = aggregate_content(per_sample, iou_threshold=args.iou_threshold)
    report_path = write_content_report(
        config.EVALUATION / f"phase1b_content_{args.run_id}.json", summary
    )
    print(f"completed_in_manifest={len(rows)} evaluated={len(per_sample)} "
          f"missing_predictions={missing_pred} missing_gt_filled={missing_gt}")
    print(f"content report -> {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
