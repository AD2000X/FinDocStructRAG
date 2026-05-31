"""Content metric tests (CPU, synthetic) - DESIGN_SPEC 6.2, Phase 1B.

Lock spatial (bbox-IoU) alignment, the three content metrics, and coverage with hand-
built gt_filled / ocr_filled tables. The key case is TATR over-segmentation, where grid-
index alignment would compare the wrong cells but IoU alignment does not.
"""

from src.eval_content import aggregate_content, content_sample_counts


def _cell(r, c, text, bbox):
    return {"row_start": r, "row_end": r + 1, "col_start": c, "col_end": c + 1,
            "text": text, "bbox": bbox}


def _table(cells):
    return {"num_rows": 2, "num_cols": 2, "cells": cells}


def test_perfect_match():
    cells = [_cell(0, 0, "Revenue", [0, 0, 50, 50]),
             _cell(0, 1, "100", [50, 0, 100, 50]),
             _cell(1, 0, "Cost", [0, 50, 50, 100]),
             _cell(1, 1, "50", [50, 50, 100, 100])]
    # distinct cell objects with identical geometry/text.
    pred = _table([dict(c) for c in cells])
    s = aggregate_content([content_sample_counts(pred, _table(cells))])
    assert s["alignment_coverage"] == 1.0
    assert s["mean_alignment_iou"] == 1.0
    assert s["cell_text_exact_match"] == 1.0
    assert s["numeric_cell_relaxed_match"] == 1.0
    assert s["non_empty_cell_content_f1"] == 1.0


def test_numeric_relaxed_catches_ocr_digit_substitution():
    gt = _table([_cell(0, 0, "Net income", [0, 0, 60, 20]),
                 _cell(0, 1, "50", [60, 0, 100, 20])])
    pred = _table([_cell(0, 0, "Net income", [0, 0, 60, 20]),
                   _cell(0, 1, "5O", [60, 0, 100, 20])])  # O instead of 0
    s = aggregate_content([content_sample_counts(pred, gt)])
    assert s["numeric_cell_relaxed_match"] == 1.0
    assert s["cell_text_exact_match"] == 0.5  # "5O" != "50"


def test_bbox_alignment_handles_over_segmentation():
    # GT: 2 stacked rows. Pred: the middle is over-segmented into 3 rows, so the GT row
    # indices no longer line up with pred row indices.
    gt = {"num_rows": 2, "num_cols": 1, "cells": [
        _cell(0, 0, "header", [0, 0, 100, 50]),
        _cell(1, 0, "data", [0, 50, 100, 100])]}
    pred = {"num_rows": 3, "num_cols": 1, "cells": [
        _cell(0, 0, "header", [0, 0, 100, 30]),
        _cell(1, 0, "X", [0, 30, 100, 60]),       # spurious over-segmented sliver
        _cell(2, 0, "data", [0, 60, 100, 100])]}
    s = aggregate_content([content_sample_counts(pred, gt)])
    # GT(0,0) -> pred row0 (IoU 0.6), GT(1,0) -> pred row2 (IoU 0.8): both exact.
    # Index alignment would have paired GT(1,0) with pred(1,0)="X" and scored 0.5.
    assert s["matched_cells"] == 2
    assert s["cell_text_exact_match"] == 1.0
    # the unmatched "X" sliver is spurious predicted text -> precision penalty (2/3).
    assert s["non_empty_precision"] == 2 / 3


def test_unmatched_gt_lowers_coverage_and_recall():
    gt = _table([_cell(0, 0, "a", [0, 0, 50, 50]),
                 _cell(0, 1, "b", [50, 0, 100, 50])])
    # prediction only covers the left cell; the right GT cell has no overlapping pred.
    pred = _table([_cell(0, 0, "a", [0, 0, 50, 50])])
    s = aggregate_content([content_sample_counts(pred, gt)])
    assert s["matched_cells"] == 1
    assert s["alignment_coverage"] == 0.5
    assert s["non_empty_recall"] == 0.5  # "b" content was lost (fn)
    assert s["cell_text_exact_match"] == 1.0  # over the one matched pair


def test_iou_threshold_param():
    gt = _table([_cell(0, 0, "x", [0, 0, 100, 100])])
    pred = _table([_cell(0, 0, "x", [0, 0, 100, 60])])  # IoU 0.6 with GT
    assert aggregate_content(
        [content_sample_counts(pred, gt, iou_threshold=0.5)])["matched_cells"] == 1
    assert aggregate_content(
        [content_sample_counts(pred, gt, iou_threshold=0.7)])["matched_cells"] == 0


def test_zero_denominator_metrics_are_null():
    gt = _table([_cell(0, 0, "", [0, 0, 50, 50])])
    pred = _table([_cell(0, 0, "", [0, 0, 50, 50])])
    s = aggregate_content([content_sample_counts(pred, gt)])
    assert s["numeric_cell_relaxed_match"] is None
    assert s["cell_text_exact_match"] is None
    assert s["alignment_coverage"] == 1.0


def test_aggregate_sums_across_samples():
    gt1 = _table([_cell(0, 0, "100", [0, 0, 50, 50])])
    pred1 = _table([_cell(0, 0, "100", [0, 0, 50, 50])])
    gt2 = _table([_cell(0, 0, "200", [0, 0, 50, 50])])
    pred2 = _table([_cell(0, 0, "999", [0, 0, 50, 50])])  # wrong number
    s = aggregate_content([
        content_sample_counts(pred1, gt1),
        content_sample_counts(pred2, gt2),
    ])
    assert s["num_samples"] == 2
    assert s["numeric_cell_relaxed_match"] == 0.5
