"""Content metric tests (CPU, synthetic) - DESIGN_SPEC 6.2, Phase 1B.

Two spatial alignment modes: aggregate (GT cell vs all pred cells centered inside it,
robust to TATR over-segmentation) and one-to-one (GT cell vs its max-IoU pred cell,
which also penalises fragmentation). Both ignore grid (row,col) indices.
"""

from src.eval_content import (
    aggregate_content_1to1,
    aggregate_content_aggregate,
    content_sample_counts_1to1,
    content_sample_counts_aggregate,
)


def _cell(r, c, text, bbox):
    return {"row_start": r, "row_end": r + 1, "col_start": c, "col_end": c + 1,
            "text": text, "bbox": bbox}


def _table(cells, num_rows=2, num_cols=2):
    return {"num_rows": num_rows, "num_cols": num_cols, "cells": cells}


# --- one-to-one mode -------------------------------------------------------

def test_1to1_perfect_match():
    cells = [_cell(0, 0, "Revenue", [0, 0, 50, 50]),
             _cell(0, 1, "100", [50, 0, 100, 50])]
    s = aggregate_content_1to1(
        [content_sample_counts_1to1(_table([dict(c) for c in cells]), _table(cells))])
    assert s["alignment_coverage"] == 1.0
    assert s["mean_alignment_iou"] == 1.0
    assert s["cell_text_exact_match"] == 1.0
    assert s["numeric_cell_relaxed_match"] == 1.0


def test_1to1_iou_threshold_param():
    gt = _table([_cell(0, 0, "x", [0, 0, 100, 100])])
    pred = _table([_cell(0, 0, "x", [0, 0, 100, 60])])  # IoU 0.6
    assert aggregate_content_1to1(
        [content_sample_counts_1to1(pred, gt, 0.5)])["matched_cells"] == 1
    assert aggregate_content_1to1(
        [content_sample_counts_1to1(pred, gt, 0.7)])["matched_cells"] == 0


def test_1to1_penalises_over_segmentation():
    # GT cell holds "Total 100"; TATR split it into "Total" and "100".
    gt = _table([_cell(0, 0, "Total 100", [0, 0, 100, 100])], num_rows=1, num_cols=1)
    pred = _table([_cell(0, 0, "Total", [0, 0, 100, 50]),
                   _cell(1, 0, "100", [0, 50, 100, 100])], num_rows=2, num_cols=1)
    s = aggregate_content_1to1([content_sample_counts_1to1(pred, gt)])
    # max-IoU picks one half -> exact match fails (this is the fragmentation penalty).
    assert s["cell_text_exact_match"] == 0.0


# --- aggregate mode --------------------------------------------------------

def test_aggregate_merges_over_segmented_text():
    # same case as above; aggregate gathers both pred cells -> recovers "Total 100".
    gt = _table([_cell(0, 0, "Total 100", [0, 0, 100, 100])], num_rows=1, num_cols=1)
    pred = _table([_cell(0, 0, "Total", [0, 0, 100, 50]),
                   _cell(1, 0, "100", [0, 50, 100, 100])], num_rows=2, num_cols=1)
    s = aggregate_content_aggregate([content_sample_counts_aggregate(pred, gt)])
    assert s["matched_cells"] == 1
    assert s["fragmented_gt_cells"] == 1
    assert s["mean_pred_cells_per_gt_cell"] == 2.0
    assert s["cell_text_exact_match"] == 1.0


def test_aggregate_sorts_reading_order():
    # pred fragments supplied out of order must join top-to-bottom.
    gt = _table([_cell(0, 0, "A B", [0, 0, 100, 100])], num_rows=1, num_cols=1)
    pred = _table([_cell(0, 0, "B", [0, 50, 100, 100]),   # lower
                   _cell(0, 0, "A", [0, 0, 100, 50])],    # upper
                  num_rows=1, num_cols=1)
    s = aggregate_content_aggregate([content_sample_counts_aggregate(pred, gt)])
    assert s["cell_text_exact_match"] == 1.0


def test_aggregate_numeric_relaxed_after_merge():
    # "$" and "1,234" split into two pred cells; merged "$ 1,234" matches "$1,234".
    gt = _table([_cell(0, 0, "$1,234", [0, 0, 100, 100])], num_rows=1, num_cols=1)
    pred = _table([_cell(0, 0, "$", [0, 0, 30, 100]),
                   _cell(0, 0, "1,234", [30, 0, 100, 100])], num_rows=1, num_cols=1)
    s = aggregate_content_aggregate([content_sample_counts_aggregate(pred, gt)])
    assert s["numeric_cell_relaxed_match"] == 1.0


def test_aggregate_far_outside_pred_not_gathered():
    gt = _table([_cell(0, 0, "x", [0, 0, 50, 50])], num_rows=1, num_cols=1)
    # a pred cell whose center is outside the GT cell must not pollute it, and counts
    # as spurious text (fp -> precision penalty).
    pred = _table([_cell(0, 0, "x", [0, 0, 50, 50]),
                   _cell(0, 0, "junk", [200, 200, 250, 250])], num_rows=1, num_cols=1)
    s = aggregate_content_aggregate([content_sample_counts_aggregate(pred, gt)])
    assert s["cell_text_exact_match"] == 1.0
    assert s["non_empty_precision"] == 0.5  # 1 tp, 1 spurious fp


def test_aggregate_unmatched_gt_lowers_coverage_and_recall():
    gt = _table([_cell(0, 0, "a", [0, 0, 50, 50]),
                 _cell(0, 1, "b", [50, 0, 100, 50])])
    pred = _table([_cell(0, 0, "a", [0, 0, 50, 50])])  # right cell not recovered
    s = aggregate_content_aggregate([content_sample_counts_aggregate(pred, gt)])
    assert s["alignment_coverage"] == 0.5
    assert s["non_empty_recall"] == 0.5


def test_aggregate_zero_denominator_metrics_are_null():
    gt = _table([_cell(0, 0, "", [0, 0, 50, 50])], num_rows=1, num_cols=1)
    pred = _table([_cell(0, 0, "", [0, 0, 50, 50])], num_rows=1, num_cols=1)
    s = aggregate_content_aggregate([content_sample_counts_aggregate(pred, gt)])
    assert s["cell_text_exact_match"] is None
    assert s["numeric_cell_relaxed_match"] is None
