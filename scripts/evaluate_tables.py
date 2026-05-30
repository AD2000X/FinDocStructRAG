"""Recompute Phase 1A topology metrics from persisted predictions (CPU, no GPU).

Reads a run's manifest, takes every sample whose final status is success, re-derives GT
topology from its structure XML (the manifest's input_path), loads the persisted
prediction (output_path), and aggregates topology metrics over ALL of them. The report
is therefore correct no matter how many resume sessions produced the predictions -
unlike the GPU runner, which only scores what it processed in a single run.

    python scripts/evaluate_tables.py --run-id smoke

Safe to re-run; it only reads predictions + annotations and rewrites the report. The
heavy parsing/metrics it calls are unit-tested in tests/ (P3); this script is the glue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.eval_table import (  # noqa: E402
    aggregate_topology,
    topology_sample_metrics,
    write_topology_report,
)
from src.fintabnet_loader import parse_structure_xml  # noqa: E402
from src.run_manifest import read_completed  # noqa: E402
from src.tatr_postprocess import normalize_tatr_prediction  # noqa: E402

PHASE = "phase1a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="smoke")
    args = ap.parse_args()

    manifest_path = config.MANIFESTS / f"{PHASE}_{args.run_id}.csv"
    rows = read_completed(manifest_path)
    if not rows:
        raise SystemExit(f"no completed samples in {manifest_path}")

    per_sample: list[dict] = []
    missing = 0
    for row in rows:
        pred_path = Path(row["output_path"])
        if not pred_path.exists():
            missing += 1
            continue
        gt = normalize_tatr_prediction(parse_structure_xml(row["input_path"]))
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        per_sample.append(topology_sample_metrics(pred, gt))

    summary = aggregate_topology(per_sample)
    report_path = write_topology_report(
        config.EVALUATION / f"{PHASE}_topology_{args.run_id}.json", summary
    )
    print(f"completed_in_manifest={len(rows)} evaluated={len(per_sample)} "
          f"missing_predictions={missing}")
    print(f"topology report -> {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
