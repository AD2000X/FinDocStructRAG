"""TATR post-processing interface (DESIGN_SPEC §5).

Phase 0 locks only the signatures Phase 1A depends on, so the canonical return type is
fixed before implementation. The bodies are implemented in Phase 1A (the design spec
gives complete reference implementations for the spanning-cell and HTML functions).
OCR / export / DataFrame functions are added in their own phases.
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
    """Derive cell bboxes from row x column intersections.

    cell_bbox = (col.x1, row.y1, col.x2, row.y2). Spanning cells override via
    apply_spanning_cells(). Returns a list of CanonicalCell-shaped dicts.
    """
    raise NotImplementedError("Phase 1A")


def validate_grid_geometry(
    row_boxes: list[dict],
    col_boxes: list[dict],
    cells: list[dict],
    logger: Optional[FailureLogger] = None,
) -> bool:
    """Sanity-check a grid: negative dims, sort order, adjacent overlap, tiny cells."""
    raise NotImplementedError("Phase 1A")


def map_spanning_bbox_to_grid(
    spanning_bbox: list[float],
    rows: list[dict],
    cols: list[dict],
    overlap_threshold: float = 0.5,
) -> Optional[dict]:
    """Map a predicted spanning-cell bbox to grid coordinates by overlap ratio.

    Returns {row_start, row_end, col_start, col_end, bbox} or None if no row/col
    meets the threshold. spanning_cell_detection_rate depends on this.
    """
    raise NotImplementedError("Phase 1A")


def apply_spanning_cells(
    cells: list[dict],
    spanning_cells: list[dict],
    rows: list[dict],
    cols: list[dict],
) -> list[dict]:
    """Map each spanning bbox to grid coords, remove covered cells, insert the merge."""
    raise NotImplementedError("Phase 1A")


def html_to_canonical(html_str: str) -> CanonicalTable:
    """Occupancy-aware HTML table parser (handles rowspan/colspan)."""
    raise NotImplementedError("Phase 1A")


def normalize_table_annotation(annotation: dict) -> CanonicalTable:
    """FinTabNet.c GT annotation -> canonical schema."""
    raise NotImplementedError("Phase 1A")


def normalize_tatr_prediction(prediction: dict) -> CanonicalTable:
    """TATR prediction -> canonical schema (same shape as the GT path)."""
    raise NotImplementedError("Phase 1A")
