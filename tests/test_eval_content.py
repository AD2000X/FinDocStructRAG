"""Content metric tests (CPU, synthetic) - DESIGN_SPEC 6.2, Phase 1B.

Lock alignment-by-anchor, the three content metrics, and coverage with hand-built
gt_filled / ocr_filled tables.
"""

from src.eval_content import (
    aggregate_content,
    content_sample_counts,
)


def _cell(r, c, text):
    return {"row_start": r, "row_end": r + 1, "col_start": c, "col_end": c + 1,
            "text": text}


def _table(cells):
    return {"num_rows": 2, "num_cols": 2, "cells": cells}


def test_perfect_match():
    cells = [_cell(0, 0, "Revenue"), _cell(0, 1, "100"),
             _cell(1, 0, "Cost"), _cell(1, 1, "50")]
    counts = content_sample_counts(_table(cells), _table(cells))
    s = aggregate_content([counts])
    assert s["alignment_coverage"] == 1.0
    assert s["cell_text_exact_match"] == 1.0
    assert s["numeric_cell_relaxed_match"] == 1.0
    assert s["non_empty_cell_content_f1"] == 1.0


def test_numeric_relaxed_catches_ocr_digit_substitution():
    gt = _table([_cell(0, 0, "Net income"), _cell(0, 1, "50")])
    # OCR misread the 0 as O: exact match fails, numeric relaxed match passes.
    pred = _table([_cell(0, 0, "Net income"), _cell(0, 1, "5O")])
    s = aggregate_content([content_sample_counts(pred, gt)])
    assert s["numeric_cell_relaxed_match"] == 1.0
    # "Net income" exact (1/1 non-empty), "5O" != "50" so the numeric cell misses exact.
    assert s["cell_text_exact_match"] == 0.5


def test_alignment_coverage_when_pred_missing_cells():
    gt = _table([_cell(0, 0, "a"), _cell(0, 1, "b"),
                 _cell(1, 0, "c"), _cell(1, 1, "d")])
    # prediction only reconstructed the top row (2 of 4 GT anchors align).
    pred = _table([_cell(0, 0, "a"), _cell(0, 1, "b")])
    s = aggregate_content([content_sample_counts(pred, gt)])
    assert s["aligned_cells"] == 2
    assert s["gt_cells"] == 4
    assert s["alignment_coverage"] == 0.5
    # exact match is over aligned non-empty GT cells only (2/2), not penalised for the
    # unaligned cells (those are an alignment/topology issue, reported via coverage).
    assert s["cell_text_exact_match"] == 1.0


def test_non_empty_f1_penalises_spurious_and_missing_text():
    # aligned 2x2: gt has text in (0,0) and (1,1); empty in (0,1) and (1,0).
    gt = _table([_cell(0, 0, "x"), _cell(0, 1, ""),
                 _cell(1, 0, ""), _cell(1, 1, "y")])
    # pred: (0,0) ok (tp); (0,1) spurious text (fp); (1,1) empty -> missed (fn).
    pred = _table([_cell(0, 0, "x"), _cell(0, 1, "noise"),
                   _cell(1, 0, ""), _cell(1, 1, "")])
    s = aggregate_content([content_sample_counts(pred, gt)])
    # tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5, f1=0.5
    assert s["non_empty_precision"] == 0.5
    assert s["non_empty_recall"] == 0.5
    assert s["non_empty_cell_content_f1"] == 0.5


def test_zero_denominator_metrics_are_null():
    # no numeric GT cells, no non-empty GT cells -> numeric/exact are null, not 0.
    gt = _table([_cell(0, 0, "")])
    pred = _table([_cell(0, 0, "")])
    s = aggregate_content([content_sample_counts(pred, gt)])
    assert s["numeric_cell_relaxed_match"] is None
    assert s["cell_text_exact_match"] is None
    assert s["alignment_coverage"] == 1.0


def test_aggregate_sums_across_samples():
    gt1 = _table([_cell(0, 0, "100")])
    pred1 = _table([_cell(0, 0, "100")])
    gt2 = _table([_cell(0, 0, "200")])
    pred2 = _table([_cell(0, 0, "999")])  # wrong number
    s = aggregate_content([
        content_sample_counts(pred1, gt1),
        content_sample_counts(pred2, gt2),
    ])
    assert s["num_samples"] == 2
    assert s["numeric_cell_relaxed_match"] == 0.5  # 1 of 2 numeric cells correct
