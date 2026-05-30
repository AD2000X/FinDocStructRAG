"""Table content metrics (DESIGN_SPEC 6.2, Phase 1B).

Pure-CPU comparison of an ocr_filled table (TATR grid + OCR text) against a gt_filled
table (GT grid + GT text). Cells are aligned by their (row_start, col_start) anchor;
metrics are computed only over aligned cells, and alignment coverage is reported
alongside so the scores are not read as full-crop recall (see 5.12 on out-of-grid words).

Three metrics:
  - non_empty_cell_content_f1: did text land in the cells that should have text
    (presence/alignment), independent of exact spelling;
  - cell_text_exact_match: of aligned non-empty GT cells, how often the text is exact;
  - numeric_cell_relaxed_match: of aligned numeric GT cells, how often the value matches
    within tolerance (via numeric_utils, which carries the V9 looks_numeric fix).

These run on ocr_filled vs gt_filled, never reporting gt_filled as an extraction (P4).
"""

from __future__ import annotations

import json
from pathlib import Path

from .canonical_schema import EVAL_TYPE_CONTENT
from .numeric_utils import looks_numeric, relaxed_numeric_match


def _norm(text: str) -> str:
    """Collapse whitespace; both sides are single-space word joins, so this makes the
    exact-match comparison robust to spacing without loosening it otherwise."""
    return " ".join((text or "").split())


def _anchor_map(table: dict) -> dict:
    return {(c["row_start"], c["col_start"]): c for c in table["cells"]}


def content_sample_counts(pred: dict, gt: dict) -> dict:
    """Per-sample raw counts (numerators/denominators) for content metrics.

    Counts, not ratios, so a batch aggregates correctly by summation.
    """
    gt_map = _anchor_map(gt)
    pred_map = _anchor_map(pred)
    aligned = set(gt_map) & set(pred_map)

    exact_num = exact_den = numeric_num = numeric_den = 0
    tp = fp = fn = 0
    for key in aligned:
        g = _norm(gt_map[key].get("text", ""))
        p = _norm(pred_map[key].get("text", ""))

        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1

        if g:
            exact_den += 1
            if p == g:
                exact_num += 1

        if looks_numeric(g):
            numeric_den += 1
            if relaxed_numeric_match(p, g):
                numeric_num += 1

    return {
        "aligned_cells": len(aligned),
        "gt_cells": len(gt_map),
        "pred_cells": len(pred_map),
        "exact_num": exact_num,
        "exact_den": exact_den,
        "numeric_num": numeric_num,
        "numeric_den": numeric_den,
        "nonempty_tp": tp,
        "nonempty_fp": fp,
        "nonempty_fn": fn,
    }


def aggregate_content(per_sample: list[dict]) -> dict:
    """Aggregate per-sample counts into a reportable content summary.

    Ratios whose denominator is zero across the batch are reported as null.
    """
    n = len(per_sample)

    def total(key: str) -> int:
        return sum(m[key] for m in per_sample)

    summary: dict = {"evaluation_type": EVAL_TYPE_CONTENT, "num_samples": n}
    gt_cells = total("gt_cells")
    aligned = total("aligned_cells")
    summary["gt_cells"] = gt_cells
    summary["pred_cells"] = total("pred_cells")
    summary["aligned_cells"] = aligned
    summary["alignment_coverage"] = aligned / gt_cells if gt_cells else None

    exact_den = total("exact_den")
    summary["cell_text_exact_match"] = (
        total("exact_num") / exact_den if exact_den else None
    )
    numeric_den = total("numeric_den")
    summary["numeric_cell_relaxed_match"] = (
        total("numeric_num") / numeric_den if numeric_den else None
    )

    tp, fp, fn = total("nonempty_tp"), total("nonempty_fp"), total("nonempty_fn")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    summary["non_empty_precision"] = precision
    summary["non_empty_recall"] = recall
    summary["non_empty_cell_content_f1"] = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return summary


def write_content_report(path: str | Path, summary: dict) -> Path:
    """Write the aggregate content summary as JSON (DESIGN_SPEC 18.3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
