"""Table content metrics (DESIGN_SPEC 6.2, Phase 1B).

Pure-CPU comparison of an ocr_filled table (TATR grid + OCR text) against a gt_filled
table (GT grid + GT text). Both sides carry cell bboxes in the same crop pixel space.

Two spatial alignment modes (grid (row,col) indices are NOT used - TATR over/under-
segmentation desynchronises them):

  aggregate (primary): each GT cell collects every pred cell whose center lies inside the
    GT cell bbox, joins their text in reading order, and compares. This measures content
    recovery within the GT cell region and is robust to TATR splitting one cell into
    several. Reports mean_pred_cells_per_gt_cell and fragmented_gt_cells.

  one_to_one (stricter, reported alongside): each GT cell matches its single max-IoU pred
    cell (>= iou_threshold). This additionally penalises topology fragmentation, so a gap
    between the two modes localises how much of the content loss is topology, not OCR.

In both modes exact/numeric are over matched GT cells only; the non-empty F1 also charges
lost GT content and spurious pred text. numeric uses the V9-fixed numeric_utils.

These run on ocr_filled vs gt_filled, never reporting gt_filled as an extraction (P4).
"""

from __future__ import annotations

import json
from pathlib import Path

from .canonical_schema import EVAL_TYPE_CONTENT
from .numeric_utils import looks_numeric, normalize_cell_text, relaxed_numeric_match

DEFAULT_IOU_THRESHOLD = 0.5


def _norm(text: str) -> str:
    """Normalize cell text for comparison (whitespace + dot-leader formatting)."""
    return normalize_cell_text(text)


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _point_in_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


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


def _text_counts(g: str, p: str) -> dict:
    """exact / numeric / non-empty tallies for one matched (gt_text, pred_text) pair."""
    c = {"exact_num": 0, "exact_den": 0, "numeric_num": 0, "numeric_den": 0,
         "tp": 0, "fp": 0, "fn": 0}
    if g and p:
        c["tp"] = 1
    elif g and not p:
        c["fn"] = 1
    elif p and not g:
        c["fp"] = 1
    if g:
        c["exact_den"] = 1
        if p == g:
            c["exact_num"] = 1
    if looks_numeric(g):
        c["numeric_den"] = 1
        if relaxed_numeric_match(p, g):
            c["numeric_num"] = 1
    return c


def _grid_cells(table: dict) -> list[dict]:
    return [c for c in table["cells"] if "bbox" in c]


def content_sample_counts_aggregate(pred: dict, gt: dict) -> dict:
    """Per-sample counts, aggregate mode: GT cell text vs all pred cells centered in it."""
    gt_cells = _grid_cells(gt)
    pred_cells = _grid_cells(pred)

    totals = {"exact_num": 0, "exact_den": 0, "numeric_num": 0, "numeric_den": 0,
              "nonempty_tp": 0, "nonempty_fp": 0, "nonempty_fn": 0}
    matched = fragmented = sum_pred_per_gt = 0
    used_pred: set[int] = set()

    for gc in gt_cells:
        inside = [(i, pc) for i, pc in enumerate(pred_cells)
                  if _point_in_bbox(_bbox_center(pc["bbox"]), gc["bbox"])]
        g = _norm(gc.get("text", ""))
        if inside:
            matched += 1
            sum_pred_per_gt += len(inside)
            if len(inside) >= 2:
                fragmented += 1
            for i, _ in inside:
                used_pred.add(i)
            inside.sort(key=lambda t: _bbox_center(t[1]["bbox"])[::-1])
            p = _norm(" ".join(pc.get("text", "") for _, pc in inside))
            c = _text_counts(g, p)
            for k in ("exact_num", "exact_den", "numeric_num", "numeric_den"):
                totals[k] += c[k]
            totals["nonempty_tp"] += c["tp"]
            totals["nonempty_fp"] += c["fp"]
            totals["nonempty_fn"] += c["fn"]
        elif g:
            totals["nonempty_fn"] += 1

    for i, pc in enumerate(pred_cells):
        if i not in used_pred and _norm(pc.get("text", "")):
            totals["nonempty_fp"] += 1

    return {
        "gt_cells": len(gt_cells),
        "pred_cells": len(pred_cells),
        "matched_cells": matched,
        "fragmented_cells": fragmented,
        "sum_pred_per_gt": sum_pred_per_gt,
        **totals,
    }


def content_sample_counts_1to1(
    pred: dict, gt: dict, iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> dict:
    """Per-sample counts, one-to-one mode: GT cell vs its single max-IoU pred cell."""
    gt_cells = _grid_cells(gt)
    pred_cells = _grid_cells(pred)

    totals = {"exact_num": 0, "exact_den": 0, "numeric_num": 0, "numeric_den": 0,
              "nonempty_tp": 0, "nonempty_fp": 0, "nonempty_fn": 0}
    matched = 0
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
            matched += 1
            sum_iou += best_iou
            used_pred.add(best_i)
            c = _text_counts(g, _norm(pred_cells[best_i].get("text", "")))
            for k in ("exact_num", "exact_den", "numeric_num", "numeric_den"):
                totals[k] += c[k]
            totals["nonempty_tp"] += c["tp"]
            totals["nonempty_fp"] += c["fp"]
            totals["nonempty_fn"] += c["fn"]
        elif g:
            totals["nonempty_fn"] += 1

    for i, pc in enumerate(pred_cells):
        if i not in used_pred and _norm(pc.get("text", "")):
            totals["nonempty_fp"] += 1

    return {
        "gt_cells": len(gt_cells),
        "pred_cells": len(pred_cells),
        "matched_cells": matched,
        "sum_iou": sum_iou,
        **totals,
    }


def _ratios(per_sample: list[dict]) -> dict:
    """The metrics common to both modes (exact / numeric / non-empty F1)."""
    def total(key: str) -> float:
        return sum(m[key] for m in per_sample)

    exact_den = total("exact_den")
    numeric_den = total("numeric_den")
    tp, fp, fn = total("nonempty_tp"), total("nonempty_fp"), total("nonempty_fn")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "cell_text_exact_match": total("exact_num") / exact_den if exact_den else None,
        "numeric_cell_relaxed_match": (
            total("numeric_num") / numeric_den if numeric_den else None
        ),
        "non_empty_precision": precision,
        "non_empty_recall": recall,
        "non_empty_cell_content_f1": (
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        ),
    }


def aggregate_content_aggregate(per_sample: list[dict]) -> dict:
    """Aggregate the primary (aggregate-mode) content summary."""
    def total(key: str) -> float:
        return sum(m[key] for m in per_sample)

    gt_cells = total("gt_cells")
    matched = total("matched_cells")
    summary = {
        "aggregation_mode": "pred_cells_inside_gt_bbox",
        "num_samples": len(per_sample),
        "gt_cells": int(gt_cells),
        "pred_cells": int(total("pred_cells")),
        "matched_cells": int(matched),
        "fragmented_gt_cells": int(total("fragmented_cells")),
        "alignment_coverage": matched / gt_cells if gt_cells else None,
        "mean_pred_cells_per_gt_cell": (
            total("sum_pred_per_gt") / matched if matched else None
        ),
    }
    summary.update(_ratios(per_sample))
    return summary


def aggregate_content_1to1(
    per_sample: list[dict], iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> dict:
    """Aggregate the stricter (one-to-one max-IoU) content summary."""
    def total(key: str) -> float:
        return sum(m[key] for m in per_sample)

    gt_cells = total("gt_cells")
    matched = total("matched_cells")
    summary = {
        "aggregation_mode": "one_to_one_max_iou",
        "iou_threshold": iou_threshold,
        "num_samples": len(per_sample),
        "gt_cells": int(gt_cells),
        "pred_cells": int(total("pred_cells")),
        "matched_cells": int(matched),
        "alignment_coverage": matched / gt_cells if gt_cells else None,
        "mean_alignment_iou": total("sum_iou") / matched if matched else None,
    }
    summary.update(_ratios(per_sample))
    return summary


def write_content_report(path: str | Path, report: dict) -> Path:
    """Write the content report as JSON (DESIGN_SPEC 18.3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
