"""TATR post-processing tests (DESIGN_SPEC §13).

Synthetic boxes only (no TATR). The 3x3 fixture: rows and cols each 10 units wide,
table spanning (0,0)-(30,30).
"""

from src.tatr_postprocess import (
    apply_spanning_cells,
    boxes_to_grid,
    can_convert_to_canonical,
    html_to_canonical,
    map_spanning_bbox_to_grid,
    validate_grid_geometry,
)


def _rows():
    return [
        {"bbox": [0, 0, 30, 10]},
        {"bbox": [0, 10, 30, 20]},
        {"bbox": [0, 20, 30, 30]},
    ]


def _cols():
    return [
        {"bbox": [0, 0, 10, 30]},
        {"bbox": [10, 0, 20, 30]},
        {"bbox": [20, 0, 30, 30]},
    ]


# --- spanning cell mapping (the 5 required tests) ---------------------------------

def test_map_spanning_bbox_covers_two_rows():
    # Column 0, rows 0-1.
    mapped = map_spanning_bbox_to_grid([0, 0, 10, 20], _rows(), _cols())
    assert mapped is not None
    assert mapped["row_start"] == 0
    assert mapped["row_end"] == 2
    assert mapped["row_end"] - mapped["row_start"] == 2
    assert mapped["col_start"] == 0
    assert mapped["col_end"] == 1


def test_map_spanning_bbox_covers_three_cols():
    # Row 0, all three columns.
    mapped = map_spanning_bbox_to_grid([0, 0, 30, 10], _rows(), _cols())
    assert mapped is not None
    assert mapped["col_start"] == 0
    assert mapped["col_end"] == 3
    assert mapped["col_end"] - mapped["col_start"] == 3
    assert mapped["row_start"] == 0
    assert mapped["row_end"] == 1


def test_map_spanning_bbox_no_overlap_returns_none():
    # Completely outside the table.
    assert map_spanning_bbox_to_grid([100, 100, 110, 110], _rows(), _cols()) is None


def test_apply_spanning_cells_merges_correctly():
    cells = boxes_to_grid(_rows(), _cols())
    assert len(cells) == 9
    spanning = [{"bbox": [0, 0, 10, 20]}]  # col 0, rows 0-1
    merged = apply_spanning_cells(cells, spanning, _rows(), _cols())
    # 9 - 2 covered + 1 merged = 8
    assert len(merged) == 8
    span_cells = [
        c for c in merged
        if c["row_end"] - c["row_start"] == 2 and c["col_start"] == 0
    ]
    assert len(span_cells) == 1
    assert span_cells[0]["col_end"] == 1


def test_apply_spanning_cells_removes_covered():
    cells = boxes_to_grid(_rows(), _cols())
    spanning = [{"bbox": [0, 0, 10, 20]}]
    merged = apply_spanning_cells(cells, spanning, _rows(), _cols())

    def is_unit(c, r, col):
        return (c["row_start"] == r and c["row_end"] == r + 1
                and c["col_start"] == col and c["col_end"] == col + 1)

    # The two individual cells covered by the span are gone.
    assert not any(is_unit(c, 0, 0) for c in merged)
    assert not any(is_unit(c, 1, 0) for c in merged)
    # An uncovered cell remains.
    assert any(is_unit(c, 2, 0) for c in merged)


# --- grid derivation / validation -------------------------------------------------

def test_boxes_to_grid_basic():
    cells = boxes_to_grid(_rows(), _cols())
    assert len(cells) == 9
    first = cells[0]
    assert first["row_start"] == 0 and first["col_start"] == 0
    assert first["bbox"] == [0, 0, 10, 10]


def test_validate_grid_geometry_ok():
    rows, cols = _rows(), _cols()
    cells = boxes_to_grid(rows, cols)
    assert validate_grid_geometry(rows, cols, cells) is True


def test_validate_grid_geometry_detects_tiny_cell():
    rows, cols = _rows(), _cols()
    cells = boxes_to_grid(rows, cols)
    cells.append({
        "row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
        "bbox": [0, 0, 1, 1], "text": "", "is_header": False, "words": [],
    })
    assert validate_grid_geometry(rows, cols, cells) is False


# --- HTML parsing -----------------------------------------------------------------

def test_html_to_canonical_simple():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    table = html_to_canonical(html)
    assert table["num_rows"] == 2
    assert table["num_cols"] == 2
    assert len(table["cells"]) == 4
    assert table["cells"][0]["is_header"] is True


def test_html_to_canonical_colspan():
    html = "<table><tr><td colspan='2'>Header</td></tr><tr><td>1</td><td>2</td></tr></table>"
    table = html_to_canonical(html)
    assert table["num_cols"] == 2
    span = table["cells"][0]
    assert span["col_start"] == 0 and span["col_end"] == 2


def test_html_to_canonical_rowspan_shifts_occupancy():
    # Cell A spans two rows in col 0; row 1's first td must land in col 1.
    html = (
        "<table>"
        "<tr><td rowspan='2'>A</td><td>B</td></tr>"
        "<tr><td>C</td></tr>"
        "</table>"
    )
    table = html_to_canonical(html)
    c = next(x for x in table["cells"] if x["text"] == "C")
    assert c["col_start"] == 1


# --- annotation gate --------------------------------------------------------------

def test_gate_accepts_html():
    ok, reason = can_convert_to_canonical({"html": "<table><tr><td>1</td></tr></table>"})
    assert ok is True and reason == ""


def test_gate_rejects_empty():
    ok, reason = can_convert_to_canonical({})
    assert ok is False and reason


def test_gate_rejects_bad_cell_span():
    bad = {"cells": [{"row_start": 1, "row_end": 1, "col_start": 0, "col_end": 1}]}
    ok, reason = can_convert_to_canonical(bad)
    assert ok is False
