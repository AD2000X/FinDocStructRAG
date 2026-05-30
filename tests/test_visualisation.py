"""Visualisation rendering tests (CPU, synthetic).

Lock the HTML/text/selection logic. The PIL drawing is checked only for shape/non-crash
(and skipped if Pillow is not installed locally; it runs on Colab).
"""

import pytest

from src import visualisation as vis


def _table_2x2_with_span():
    # 2 cols, top cell spans both columns of row 0; row 1 has two single cells.
    return {
        "num_rows": 2,
        "num_cols": 2,
        "cells": [
            {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 2,
             "bbox": [0, 0, 100, 25], "text": "Header", "is_header": True},
            {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1,
             "bbox": [0, 25, 50, 50], "text": "a"},
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2,
             "bbox": [50, 25, 100, 50], "text": "b"},
        ],
    }


def test_topology_to_html_spans_and_headers():
    out = vis.topology_to_html(_table_2x2_with_span())
    assert out.count("<tr>") == 2
    assert 'colspan="2"' in out
    assert "<th" in out and "Header" in out
    # the two row-1 cells render as plain td
    assert out.count("<td>a</td>") == 1
    assert out.count("<td>b</td>") == 1


def test_topology_to_html_escapes_text():
    table = {"num_rows": 1, "num_cols": 1, "cells": [
        {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1, "text": "<x>"}]}
    assert "&lt;x&gt;" in vis.topology_to_html(table)


def test_summary_to_html_has_metrics_and_note():
    summary = {"row_count_accuracy": 0.79, "num_samples": 300}
    out = vis.summary_to_html(summary, subset_note="random 300, seed 42")
    assert "row_count_accuracy" in out
    assert "0.79" in out
    assert "random 300, seed 42" in out


def test_geometry_report_lists_flags():
    art = {
        "sample_id": "s1",
        "row_boxes": [{}, {}],
        "col_boxes": [{}],
        "spanning_cells": [],
        "geometry_validation": {"valid": False,
                                "flags": ["adjacent rows overlap > 0.3"]},
    }
    report = vis.geometry_report(art)
    assert "sample_id: s1" in report
    assert "valid: False" in report
    assert "- adjacent rows overlap > 0.3" in report


def test_geometry_report_no_flags():
    art = {"sample_id": "s", "geometry_validation": {"valid": True, "flags": []}}
    assert "(none)" in vis.geometry_report(art)


def test_is_failure_candidate():
    assert vis.is_failure_candidate(
        {"geometry_validation": {"flags": ["x"]}}) is True
    assert vis.is_failure_candidate(
        {"geometry_validation": {"flags": []}}) is False
    assert vis.is_failure_candidate({}) is False


def test_is_spanning():
    assert vis.is_spanning(
        {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 2}) is True
    assert vis.is_spanning(
        {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1}) is False


def test_draw_functions_return_same_size_image():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (120, 60), "white")
    raw = {"row_boxes": [{"bbox": [0, 0, 120, 30], "score": 0.9, "label": "table row"}],
           "col_boxes": [{"bbox": [0, 0, 60, 60], "score": 0.8, "label": "table column"}]}
    table = _table_2x2_with_span()
    for out in (
        vis.draw_tatr_overlay(img, raw),
        vis.draw_cell_grid(img, table),
        vis.draw_spanning_cells(img, table),
    ):
        assert out.size == (120, 60)
        assert out is not img  # drew on a copy
