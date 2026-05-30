"""Table topology metrics (DESIGN_SPEC §6.2, Phase 1A).

Pure-CPU comparison of a predicted CanonicalTable against a GT CanonicalTable,
plus aggregation across a batch and a writer for the reportable JSON
(outputs/evaluation/phase1a_topology.json, §18.3).

Scope: topology only. Content metrics (§6.2) live with Phase 1B. header_detection_accuracy
and html_structure_match are deferred until header detection is wired in (TATR predictions
currently carry no is_header signal, see normalize_tatr_prediction).

These run on tatr_predicted/ tables, never gt_filled/ ones (P4).
"""

from __future__ import annotations

import json
from pathlib import Path

from .canonical_schema import CanonicalTable, EVAL_TYPE_TOPOLOGY


def occupancy_set(table: CanonicalTable) -> set[tuple[int, int]]:
    """Grid positions (row, col) occupied by the table's cells (half-open ranges)."""
    occ: set[tuple[int, int]] = set()
    for c in table["cells"]:
        for r in range(c["row_start"], c["row_end"]):
            for col in range(c["col_start"], c["col_end"]):
                occ.add((r, col))
    return occ


def spanning_cells_of(table: CanonicalTable) -> set[tuple[int, int, int, int]]:
    """Cells spanning more than one row or column, as (rs, re, cs, ce) tuples."""
    spans = set()
    for c in table["cells"]:
        if c["row_end"] - c["row_start"] > 1 or c["col_end"] - c["col_start"] > 1:
            spans.add((c["row_start"], c["row_end"], c["col_start"], c["col_end"]))
    return spans


def _prf(pred: set, gt: set) -> tuple[float, float, float]:
    """Precision, recall, F1 for two sets. Two empty sets score a perfect 1.0."""
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def topology_sample_metrics(
    pred: CanonicalTable, gt: CanonicalTable
) -> dict:
    """Per-sample topology metrics comparing a prediction to its GT table."""
    p, r, f1 = _prf(occupancy_set(pred), occupancy_set(gt))
    gt_spans = spanning_cells_of(gt)
    matched_spans = len(gt_spans & spanning_cells_of(pred))
    return {
        "row_count_correct": pred["num_rows"] == gt["num_rows"],
        "col_count_correct": pred["num_cols"] == gt["num_cols"],
        "cell_occupancy_precision": p,
        "cell_occupancy_recall": r,
        "cell_occupancy_f1": f1,
        "gt_spanning_cells": len(gt_spans),
        "matched_spanning_cells": matched_spans,
    }


def aggregate_topology(per_sample: list[dict]) -> dict:
    """Aggregate per-sample topology metrics into a reportable summary.

    spanning_cell_detection_rate is null when no GT spanning cells exist in the batch.
    """
    n = len(per_sample)
    summary: dict = {
        "evaluation_type": EVAL_TYPE_TOPOLOGY,
        "num_samples": n,
    }
    if n == 0:
        summary.update({
            "row_count_accuracy": 0.0,
            "col_count_accuracy": 0.0,
            "cell_occupancy_f1": 0.0,
            "spanning_cell_detection_rate": None,
        })
        return summary

    summary["row_count_accuracy"] = sum(
        m["row_count_correct"] for m in per_sample
    ) / n
    summary["col_count_accuracy"] = sum(
        m["col_count_correct"] for m in per_sample
    ) / n
    summary["cell_occupancy_f1"] = sum(
        m["cell_occupancy_f1"] for m in per_sample
    ) / n

    total_gt_spans = sum(m["gt_spanning_cells"] for m in per_sample)
    total_matched = sum(m["matched_spanning_cells"] for m in per_sample)
    summary["spanning_cell_detection_rate"] = (
        total_matched / total_gt_spans if total_gt_spans else None
    )
    return summary


def write_topology_report(path: str | Path, summary: dict) -> Path:
    """Write the aggregate topology summary as JSON (DESIGN_SPEC §18.3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
