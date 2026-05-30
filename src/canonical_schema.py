"""Canonical table schema.

Cross-cutting contract: normalize_table_annotation(), normalize_tatr_prediction(),
and html_to_canonical() must all emit this shape, otherwise evaluation will not line
up. Fixed in Phase 0 so downstream phases share one definition.

TypedDict is used (not a dataclass) so the existing dict-returning functions in the
design spec satisfy the schema without conversion.
"""

from __future__ import annotations

from typing import TypedDict


# Allowed metadata values (see DESIGN_SPEC §5.7). Kept as constants so producers and
# evaluators agree on the strings used to separate GT-filled from extraction outputs.
TEXT_SOURCE_GT = "gt"
TEXT_SOURCE_OCR = "ocr"
TEXT_SOURCE_NONE = "none"
TEXT_SOURCES = (TEXT_SOURCE_GT, TEXT_SOURCE_OCR, TEXT_SOURCE_NONE)

EVAL_TYPE_TOPOLOGY = "topology"
EVAL_TYPE_CONTENT = "content"
EVAL_TYPES = (EVAL_TYPE_TOPOLOGY, EVAL_TYPE_CONTENT)


class CanonicalCell(TypedDict, total=False):
    """A single table cell.

    row/col indices are half-open: a 1x1 cell at (r, c) has row_start=r,
    row_end=r+1, col_start=c, col_end=c+1. Spanning cells widen the range.

    Required: row_start, row_end, col_start, col_end, text, is_header.
    Optional: bbox (present for TATR-derived cells), words (present after OCR).
    """

    row_start: int
    row_end: int
    col_start: int
    col_end: int
    text: str
    is_header: bool
    bbox: list[float]
    words: list[dict]


class CanonicalTable(TypedDict, total=False):
    """A reconstructed table.

    Required: num_rows, num_cols, cells.
    Optional: meta (text_source / evaluation_type provenance).
    """

    num_rows: int
    num_cols: int
    cells: list[CanonicalCell]
    meta: TableMeta


class TableMeta(TypedDict, total=False):
    """Provenance for a canonical table (DESIGN_SPEC §5.7).

    text_source distinguishes GT-filled tables (QA pipeline validation only) from
    OCR-filled tables (real extraction outputs). evaluation_type separates topology
    metrics from content metrics. P4: GT-filled is never reported as extraction.
    """

    sample_id: str
    text_source: str       # one of TEXT_SOURCES
    evaluation_type: str   # one of EVAL_TYPES
