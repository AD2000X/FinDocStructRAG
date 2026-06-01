"""Recompute Phase 1B content metrics from persisted ocr_filled vs gt_filled (CPU).

Reads the ocr_filled run manifest, takes every sample whose final status is success,
loads its ocr_filled table (the manifest's output_path) and the matching gt_filled table
(outputs/tables/gt_filled/<sample_id>.json), and aggregates content metrics over all
pairs. Like evaluate_tables.py, scoring runs over every persisted pair regardless of how
many sessions produced them.

    python scripts/evaluate_content.py --run-id debug

Emits three views (see src/eval_content): the primary aggregate metric (content recovery
within GT cell regions), the stricter one-to-one metric (also penalises topology
fragmentation), and a topology-matched subset (samples whose row/col counts match GT, to
read OCR quality where topology is correct). Safe to re-run; only reads tables. The
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
    aggregate_content_1to1,
    aggregate_content_aggregate,
    content_sample_counts_1to1,
    content_sample_counts_aggregate,
    write_content_report,
)
from src.run_manifest import read_completed  # noqa: E402

OCR_PHASE = "phase1b_ocr"


def _topology_matches(pred: dict, gt: dict) -> bool:
    return (pred.get("num_rows") == gt.get("num_rows")
            and pred.get("num_cols") == gt.get("num_cols"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="debug", help="the ocr_filled run-id")
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="min bbox IoU for the one-to-one match")
    args = ap.parse_args()

    manifest_path = config.MANIFESTS / f"{OCR_PHASE}_{args.run_id}.csv"
    rows = read_completed(manifest_path)
    if not rows:
        raise SystemExit(f"no completed samples in {manifest_path}")

    agg_counts: list[dict] = []
    one_counts: list[dict] = []
    topo_agg_counts: list[dict] = []
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
        ac = content_sample_counts_aggregate(pred, gt)
        agg_counts.append(ac)
        one_counts.append(
            content_sample_counts_1to1(pred, gt, iou_threshold=args.iou_threshold))
        if _topology_matches(pred, gt):
            topo_agg_counts.append(ac)

    report = {
        "evaluation_type": "content",
        "num_samples": len(agg_counts),
        "aggregate": aggregate_content_aggregate(agg_counts),
        "one_to_one": aggregate_content_1to1(
            one_counts, iou_threshold=args.iou_threshold),
        "topology_matched_subset": {
            "num_samples": len(topo_agg_counts),
            "metrics": (
                aggregate_content_aggregate(topo_agg_counts)
                if topo_agg_counts else None
            ),
        },
    }
    report_path = write_content_report(
        config.EVALUATION / f"phase1b_content_{args.run_id}.json", report
    )
    print(f"completed_in_manifest={len(rows)} evaluated={len(agg_counts)} "
          f"missing_predictions={missing_pred} missing_gt_filled={missing_gt} "
          f"topology_matched={len(topo_agg_counts)}")
    print(f"content report -> {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
