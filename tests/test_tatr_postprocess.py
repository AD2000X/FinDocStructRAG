"""TATR post-processing tests (DESIGN_SPEC §13).

Synthetic boxes only (no TATR). The 3x3 fixture: rows and cols each 10 units wide,
table spanning (0,0)-(30,30).
"""

from src.tatr_postprocess import (
    apply_spanning_cells,
    boxes_to_grid,
    can_convert_to_canonical,
    dedup_row_col_bands,
    html_to_canonical,
    map_spanning_bbox_to_grid,
    normalize_tatr_prediction,
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


def test_normalize_marks_column_headers():
    # 3x3 grid; a column_headers box spans the top row across the full width.
    pred = {
        "row_boxes": _rows(),
        "col_boxes": _cols(),
        "column_headers": [{"bbox": [0, 0, 30, 10]}],
    }
    table = normalize_tatr_prediction(pred)
    header = [c for c in table["cells"] if c["row_start"] == 0]
    body = [c for c in table["cells"] if c["row_start"] > 0]
    assert header and all(c["is_header"] for c in header)
    assert body and not any(c["is_header"] for c in body)


def test_normalize_without_column_headers_marks_nothing():
    table = normalize_tatr_prediction({"row_boxes": _rows(), "col_boxes": _cols()})
    assert not any(c["is_header"] for c in table["cells"])


def test_normalize_marks_header_when_box_misses_cell_center():
    # Real-data regression: the column-header box is a narrow band at the top of row 0
    # (y 0-4), so it does NOT contain row 0's cell center (y=5) - the old center-in-box
    # test marked nothing. Overlap (IoMin) must still flag the row-0 cells as headers.
    rows = [{"bbox": [0, 0, 30, 10]}, {"bbox": [0, 10, 30, 20]}]
    cols = [{"bbox": [0, 0, 30, 20]}]
    pred = {"row_boxes": rows, "col_boxes": cols,
            "column_headers": [{"bbox": [0, 0, 30, 4]}]}
    table = normalize_tatr_prediction(pred)
    row0 = [c for c in table["cells"] if c["row_start"] == 0]
    row1 = [c for c in table["cells"] if c["row_start"] == 1]
    assert row0 and all(c["is_header"] for c in row0)
    assert row1 and not any(c["is_header"] for c in row1)


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


# --- dedup_row_col_bands ---


def _b(lo, hi, score=0.9, axis=1):
    """Make a synthetic band box. axis=1 -> row (y), axis=0 -> col (x)."""
    if axis == 1:
        return {"bbox": [0.0, float(lo), 30.0, float(hi)], "score": score}
    return {"bbox": [float(lo), 0.0, float(hi), 30.0], "score": score}


def test_dedup_no_overlap_keeps_all():
    rows = [_b(0, 10), _b(15, 25), _b(30, 40)]
    result = dedup_row_col_bands({"row_boxes": rows, "col_boxes": []})
    assert len(result["row_boxes"]) == 3


def test_dedup_overlap_keeps_higher_score():
    # Row A [0,10] score=0.5 overlaps Row B [5,15] score=0.9 -> keep B
    rows = [_b(0, 10, score=0.5), _b(5, 15, score=0.9)]
    result = dedup_row_col_bands({"row_boxes": rows, "col_boxes": []})
    kept = result["row_boxes"]
    assert len(kept) == 1
    assert kept[0]["score"] == 0.9


def test_dedup_overlap_keeps_lower_start_when_score_wins():
    # Row A [0,10] score=0.9 overlaps Row B [5,15] score=0.3 -> keep A
    rows = [_b(0, 10, score=0.9), _b(5, 15, score=0.3)]
    result = dedup_row_col_bands({"row_boxes": rows, "col_boxes": []})
    kept = result["row_boxes"]
    assert len(kept) == 1
    assert kept[0]["score"] == 0.9


def test_dedup_chain_three_overlapping():
    # A [0,10] score=0.9, B [5,15] score=0.3, C [13,23] score=0.8
    # A vs B: overlap=5, smaller=10, ratio=0.5 > 0.3 -> keep A (higher score)
    # kept[-1]=A vs C: overlap=max(0,min(10,23)-max(0,13))=0 -> keep C
    # Result: [A, C]
    rows = [_b(0, 10, score=0.9), _b(5, 15, score=0.3), _b(13, 23, score=0.8)]
    result = dedup_row_col_bands({"row_boxes": rows, "col_boxes": []})
    assert len(result["row_boxes"]) == 2
    scores = {r["score"] for r in result["row_boxes"]}
    assert scores == {0.9, 0.8}


def test_dedup_col_axis():
    # Two overlapping col boxes (x-axis)
    cols = [_b(0, 10, score=0.4, axis=0), _b(6, 16, score=0.7, axis=0)]
    result = dedup_row_col_bands({"row_boxes": [], "col_boxes": cols})
    kept = result["col_boxes"]
    assert len(kept) == 1
    assert kept[0]["score"] == 0.7


def test_dedup_no_overlap_at_exact_threshold():
    # overlap=3, smaller=10 -> ratio=0.3 (NOT > 0.3) -> keep both
    rows = [_b(0, 10, score=0.9), _b(7, 17, score=0.8)]
    result = dedup_row_col_bands({"row_boxes": rows, "col_boxes": []})
    assert len(result["row_boxes"]) == 2


def test_dedup_other_keys_passthrough():
    pred = {
        "row_boxes": [_b(0, 10)],
        "col_boxes": [_b(0, 10, axis=0)],
        "spanning_cells": [{"bbox": [0, 0, 10, 10]}],
        "column_headers": [{"bbox": [0, 0, 30, 5]}],
    }
    result = dedup_row_col_bands(pred)
    assert result["spanning_cells"] is pred["spanning_cells"]
    assert result["column_headers"] is pred["column_headers"]


def test_dedup_makes_previously_invalid_grid_valid():
    # Two rows heavily overlapping -> validate fails before dedup, passes after
    rows = [_b(0, 20, score=0.9), _b(5, 25, score=0.5)]
    cols = [_b(0, 10, axis=0), _b(10, 20, axis=0), _b(20, 30, axis=0)]
    pred = {"row_boxes": rows, "col_boxes": cols}

    canonical_before = normalize_tatr_prediction(pred)
    valid_before = validate_grid_geometry(
        sorted(pred["row_boxes"], key=lambda r: r["bbox"][1]),
        sorted(pred["col_boxes"], key=lambda c: c["bbox"][0]),
        canonical_before["cells"],
    )
    assert not valid_before

    deduped = dedup_row_col_bands(pred)
    canonical_after = normalize_tatr_prediction(deduped)
    valid_after = validate_grid_geometry(
        sorted(deduped["row_boxes"], key=lambda r: r["bbox"][1]),
        sorted(deduped["col_boxes"], key=lambda c: c["bbox"][0]),
        canonical_after["cells"],
    )
    assert valid_after
