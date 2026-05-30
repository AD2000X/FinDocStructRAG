"""TATR post-processing (DESIGN_SPEC §5).

Pure-CPU logic: derive a cell grid from row/column boxes, map predicted spanning
cells back to grid coordinates, validate grid geometry, and parse GT HTML into the
canonical schema. GPU inference (the TATR model call itself) lives in the Colab runner;
everything here is unit-testable with synthetic boxes.
"""

from __future__ import annotations

from typing import Optional

from .canonical_schema import CanonicalTable
from .failure_logger import FailureLogger


def boxes_to_grid(
    row_boxes: list[dict],
    col_boxes: list[dict],
    spanning_cells: Optional[list[dict]] = None,
) -> list[dict]:
    """Derive cell bboxes from row x column intersections (DESIGN_SPEC §5.2).

    cell_bbox = (col.x1, row.y1, col.x2, row.y2). Spanning cells override via
    apply_spanning_cells().
    """
    rows = sorted(row_boxes, key=lambda r: r["bbox"][1])
    cols = sorted(col_boxes, key=lambda c: c["bbox"][0])

    cells = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            cells.append({
                "row_start": i, "row_end": i + 1,
                "col_start": j, "col_end": j + 1,
                "bbox": [col["bbox"][0], row["bbox"][1],
                         col["bbox"][2], row["bbox"][3]],
                "text": "", "is_header": False, "words": []
            })

    if spanning_cells:
        cells = apply_spanning_cells(cells, spanning_cells, rows, cols)

    return cells


def validate_grid_geometry(
    row_boxes: list[dict],
    col_boxes: list[dict],
    cells: list[dict],
    logger: Optional[FailureLogger] = None,
    sample_id: str = "unknown",
) -> bool:
    """Sanity-check a grid (DESIGN_SPEC §5.3).

    Checks: negative dimensions, row/col sort order, adjacent overlap > 0.3, and
    tiny cells (area < 100). Returns True when the grid is sane; failures are logged
    if a FailureLogger is given.
    """
    ok = True

    def fail(reason: str) -> None:
        nonlocal ok
        ok = False
        if logger is not None:
            logger.log(sample_id, "phase1a", "grid_geometry", reason)

    for r in row_boxes:
        x1, y1, x2, y2 = r["bbox"]
        if x2 <= x1 or y2 <= y1:
            fail("row box has non-positive dimensions")
    for c in col_boxes:
        x1, y1, x2, y2 = c["bbox"]
        if x2 <= x1 or y2 <= y1:
            fail("col box has non-positive dimensions")

    ys = [r["bbox"][1] for r in row_boxes]
    if ys != sorted(ys):
        fail("row boxes not sorted top-to-bottom")
    xs = [c["bbox"][0] for c in col_boxes]
    if xs != sorted(xs):
        fail("col boxes not sorted left-to-right")

    # Adjacent rows/cols should not overlap by more than 0.3 of the smaller extent.
    srows = sorted(row_boxes, key=lambda r: r["bbox"][1])
    for a, b in zip(srows, srows[1:]):
        ay1, ay2 = a["bbox"][1], a["bbox"][3]
        by1, by2 = b["bbox"][1], b["bbox"][3]
        overlap = max(0, min(ay2, by2) - max(ay1, by1))
        smaller = min(ay2 - ay1, by2 - by1)
        if smaller > 0 and overlap / smaller > 0.3:
            fail("adjacent rows overlap > 0.3")
    scols = sorted(col_boxes, key=lambda c: c["bbox"][0])
    for a, b in zip(scols, scols[1:]):
        ax1, ax2 = a["bbox"][0], a["bbox"][2]
        bx1, bx2 = b["bbox"][0], b["bbox"][2]
        overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        smaller = min(ax2 - ax1, bx2 - bx1)
        if smaller > 0 and overlap / smaller > 0.3:
            fail("adjacent cols overlap > 0.3")

    for c in cells:
        if "bbox" in c:
            x1, y1, x2, y2 = c["bbox"]
            if (x2 - x1) * (y2 - y1) < 100:
                fail("tiny cell area < 100")

    return ok


def map_spanning_bbox_to_grid(
    spanning_bbox: list[float],
    rows: list[dict],
    cols: list[dict],
    overlap_threshold: float = 0.5,
) -> Optional[dict]:
    """Convert a predicted spanning-cell bbox into grid coordinates (DESIGN_SPEC §5.4).

    A row/col is covered if overlap_length / row_or_col_length >= overlap_threshold.
    Returns {row_start, row_end, col_start, col_end, bbox} (ends exclusive) or None if
    no row/col meets the threshold.
    """
    sx1, sy1, sx2, sy2 = spanning_bbox

    covered_rows = []
    for i, row in enumerate(rows):
        ry1, ry2 = row["bbox"][1], row["bbox"][3]
        overlap = max(0, min(sy2, ry2) - max(sy1, ry1))
        row_height = ry2 - ry1
        if row_height > 0 and overlap / row_height >= overlap_threshold:
            covered_rows.append(i)

    covered_cols = []
    for j, col in enumerate(cols):
        cx1, cx2 = col["bbox"][0], col["bbox"][2]
        overlap = max(0, min(sx2, cx2) - max(sx1, cx1))
        col_width = cx2 - cx1
        if col_width > 0 and overlap / col_width >= overlap_threshold:
            covered_cols.append(j)

    if not covered_rows or not covered_cols:
        return None

    return {
        "row_start": min(covered_rows),
        "row_end": max(covered_rows) + 1,
        "col_start": min(covered_cols),
        "col_end": max(covered_cols) + 1,
        "bbox": spanning_bbox,
    }


def apply_spanning_cells(
    cells: list[dict],
    spanning_cells: list[dict],
    rows: list[dict],
    cols: list[dict],
) -> list[dict]:
    """Map each spanning bbox to grid coords, remove covered cells, insert the merge
    (DESIGN_SPEC §5.4)."""
    for span_box in spanning_cells:
        mapped = map_spanning_bbox_to_grid(span_box["bbox"], rows, cols)
        if mapped is None:
            continue

        cells = [c for c in cells if not (
            c["row_start"] >= mapped["row_start"] and
            c["row_end"] <= mapped["row_end"] and
            c["col_start"] >= mapped["col_start"] and
            c["col_end"] <= mapped["col_end"]
        )]

        cells.append({
            "row_start": mapped["row_start"],
            "row_end": mapped["row_end"],
            "col_start": mapped["col_start"],
            "col_end": mapped["col_end"],
            "bbox": mapped["bbox"],
            "text": "", "is_header": False, "words": []
        })

    return cells


def html_to_canonical(html_str: str) -> CanonicalTable:
    """Occupancy-aware HTML table parser (DESIGN_SPEC §5.10).

    Uses an occupancy grid so rowspan/colspan shift later cells into free columns.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_str, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return {"num_rows": 0, "num_cols": 0, "cells": []}

    occupied = set()
    cells = []
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for td in row.find_all(["td", "th"]):
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            rowspan = int(td.get("rowspan", 1))
            colspan = int(td.get("colspan", 1))
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied.add((r, c))
            cells.append({
                "row_start": row_idx, "row_end": row_idx + rowspan,
                "col_start": col_idx, "col_end": col_idx + colspan,
                "text": td.get_text(strip=True),
                "is_header": td.name == "th",
            })
            col_idx += colspan

    num_rows = len(rows)
    num_cols = max((c["col_end"] for c in cells), default=0)
    return {"num_rows": num_rows, "num_cols": num_cols, "cells": cells}


def can_convert_to_canonical(annotation: dict) -> tuple[bool, str]:
    """Front gate for FinTabNet.c annotations (DESIGN_SPEC §5.9).

    Conservative: reject anything normalize_table_annotation() cannot map, so a
    malformed sample is logged and skipped rather than corrupting metrics. Returns
    (ok, reason); reason is "" when ok.

    Note: this validates the normalized-input contract (an "html" string or a "cells"
    list of row/col-spanned dicts). The raw FinTabNet.c field -> this shape adapter is
    finalized on Colab against the real annotation format.
    """
    if not isinstance(annotation, dict):
        return False, "annotation is not a dict"

    html = annotation.get("html")
    has_html = isinstance(html, str) and bool(html.strip())
    cells = annotation.get("cells")
    has_cells = isinstance(cells, list) and len(cells) > 0

    if not (has_html or has_cells):
        return False, "annotation has neither non-empty 'html' nor 'cells'"

    if has_cells:
        for i, c in enumerate(cells):
            if not isinstance(c, dict):
                return False, f"cell {i} is not a dict"
            for k in ("row_start", "row_end", "col_start", "col_end"):
                if k not in c:
                    return False, f"cell {i} missing '{k}'"
            if c["row_end"] <= c["row_start"] or c["col_end"] <= c["col_start"]:
                return False, f"cell {i} has non-positive span"

    return True, ""


def normalize_table_annotation(annotation: dict) -> CanonicalTable:
    """FinTabNet.c GT annotation -> canonical schema.

    Gated by can_convert_to_canonical(). HTML annotations go through the occupancy-aware
    parser; cell-list annotations are normalized directly.
    """
    ok, reason = can_convert_to_canonical(annotation)
    if not ok:
        raise ValueError(f"cannot convert annotation: {reason}")

    html = annotation.get("html")
    if isinstance(html, str) and html.strip():
        return html_to_canonical(html)

    cells = [
        {
            "row_start": c["row_start"], "row_end": c["row_end"],
            "col_start": c["col_start"], "col_end": c["col_end"],
            "text": c.get("text", ""),
            "is_header": bool(c.get("is_header", False)),
        }
        for c in annotation["cells"]
    ]
    num_rows = max((c["row_end"] for c in cells), default=0)
    num_cols = max((c["col_end"] for c in cells), default=0)
    return {"num_rows": num_rows, "num_cols": num_cols, "cells": cells}


def normalize_tatr_prediction(prediction: dict) -> CanonicalTable:
    """TATR prediction -> canonical schema (same shape as the GT path).

    Expects row_boxes / col_boxes (and optional spanning_cells) as lists of dicts with
    a "bbox" key. Header detection is added with the Colab metrics step.
    """
    rows = sorted(prediction.get("row_boxes", []), key=lambda r: r["bbox"][1])
    cols = sorted(prediction.get("col_boxes", []), key=lambda c: c["bbox"][0])
    cells = boxes_to_grid(rows, cols, prediction.get("spanning_cells"))
    return {"num_rows": len(rows), "num_cols": len(cols), "cells": cells}
