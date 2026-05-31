"""Table content metrics (DESIGN_SPEC 6.2, Phase 1B).

Pure-CPU comparison of an ocr_filled table (TATR grid + OCR text) against a gt_filled
table (GT grid + GT text). Cells are aligned SPATIALLY: each GT cell is matched to the
pred cell with the largest bbox IoU, and only pairs with IoU >= iou_threshold count.
This is topology-independent - grid (row,col) indices desynchronise as soon as TATR over-
or under-segments a row, so index alignment would compare physically different cells.
Both sides carry bboxes in the same crop pixel space, so IoU matching is well defined.

Reported:
  - alignment_coverage = matched GT cells / GT cells (how much of the GT grid the
    prediction spatially recovers); mean_alignment_iou over matched pairs;
  - non_empty_cell_content_f1: did text land where text belongs (presence), counting
    lost GT content and spurious pred content;
  - cell_text_exact_match / numeric_cell_relaxed_match: over matched pairs only, so the
    text scores are read on sensible pairings (numeric via the V9-fixed numeric_utils).

These run on ocr_filled vs gt_filled, never reporting gt_filled as an extraction (P4).
"""

from __future__ import annotations

import json
from pathlib import Path

from .canonical_schema import EVAL_TYPE_CONTENT
from .numeric_utils import looks_numeric, relaxed_numeric_match

DEFAULT_IOU_THRESHOLD = 0.5


def _norm(text: str) -> str:
    """Collapse whitespace; both sides are single-space word joins, so this makes the
    exact-match comparison robust to spacing without loosening it otherwise."""
    return " ".join((text or "").split())


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def content_sample_counts(
    pred: dict, gt: dict, iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> dict:
    """Per-sample raw counts (numerators/denominators) for content metrics.

    Counts, not ratios, so a batch aggregates correctly by summation. Each GT cell is
    matched to its max-IoU pred cell (>= iou_threshold). exact/numeric are over matched
    pairs only; the non-empty F1 also charges lost GT content and spurious pred content.
    """
    gt_cells = [c for c in gt["cells"] if "bbox" in c]
    pred_cells = [c for c in pred["cells"] if "bbox" in c]

    exact_num = exact_den = numeric_num = numeric_den = 0
    tp = fp = fn = 0
    matched_count = 0
    sum_iou = 0.0
    used_pred: set[int] = set()

    for gc in gt_cells:
        best_i, best_iou = -1, 0.0
        for i, pc in enumerate(pred_cells):
            iou = _iou(gc["bbox"], pc["bbox"])
            if iou > best_iou:
                best_iou, best_i = iou, i

        g = _norm(gc.get("text", ""))
        if best_i >= 0 and best_iou >= iou_threshold:
            matched_count += 1
            sum_iou += best_iou
            used_pred.add(best_i)
            p = _norm(pred_cells[best_i].get("text", ""))

            if g and p:
                tp += 1
            elif g and not p:
                fn += 1
            elif p and not g:
                fp += 1

            if g:
                exact_den += 1
                if p == g:
                    exact_num += 1
            if looks_numeric(g):
                numeric_den += 1
                if relaxed_numeric_match(p, g):
                    numeric_num += 1
        else:
            # GT cell the prediction did not spatially recover; lost content if non-empty.
            if g:
                fn += 1

    # Spurious predicted text: a pred cell with text that matched no GT cell.
    for i, pc in enumerate(pred_cells):
        if i not in used_pred and _norm(pc.get("text", "")):
            fp += 1

    return {
        "gt_cells": len(gt_cells),
        "pred_cells": len(pred_cells),
        "matched_cells": matched_count,
        "sum_iou": sum_iou,
        "exact_num": exact_num,
        "exact_den": exact_den,
        "numeric_num": numeric_num,
        "numeric_den": numeric_den,
        "nonempty_tp": tp,
        "nonempty_fp": fp,
        "nonempty_fn": fn,
    }


def aggregate_content(
    per_sample: list[dict], iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> dict:
    """Aggregate per-sample counts into a reportable content summary.

    Ratios whose denominator is zero across the batch are reported as null.
    """
    n = len(per_sample)

    def total(key: str) -> float:
        return sum(m[key] for m in per_sample)

    summary: dict = {
        "evaluation_type": EVAL_TYPE_CONTENT,
        "num_samples": n,
        "iou_threshold": iou_threshold,
    }
    gt_cells = total("gt_cells")
    matched = total("matched_cells")
    summary["gt_cells"] = int(gt_cells)
    summary["pred_cells"] = int(total("pred_cells"))
    summary["matched_cells"] = int(matched)
    summary["alignment_coverage"] = matched / gt_cells if gt_cells else None
    summary["mean_alignment_iou"] = total("sum_iou") / matched if matched else None

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
