"""Assemble word tokens into a content-filled canonical table (Phase 1B).

gt_filled and ocr_filled are produced the same way - a cell grid plus a list of words
run through assign_words_to_cells - and differ only in their inputs (GT structure XML +
GT words vs TATR-predicted grid + OCR words). Keeping one builder makes that symmetry
explicit and means the two outputs cannot drift apart in how text is placed.

P4: text_source on the table's meta records which it is; gt_filled is never reported as
an extraction output.
"""

from __future__ import annotations

from typing import Optional

from .canonical_schema import CanonicalTable, EVAL_TYPE_CONTENT
from .failure_logger import FailureLogger
from .tatr_postprocess import assign_words_to_cells, normalize_tatr_prediction


def fill_table(
    grid_boxes: dict,
    words: list[dict],
    *,
    sample_id: str,
    text_source: str,
    logger: Optional[FailureLogger] = None,
) -> CanonicalTable:
    """Build a content table from a grid source and word tokens.

    grid_boxes: dict with row_boxes / col_boxes / spanning_cells (a parsed GT structure
    XML or a TATR prediction). words: dicts with bbox + text. text_source: 'gt' or 'ocr'.
    """
    table = normalize_tatr_prediction(grid_boxes)
    assign_words_to_cells(table["cells"], words, logger, sample_id)
    table["meta"] = {
        "sample_id": sample_id,
        "text_source": text_source,
        "evaluation_type": EVAL_TYPE_CONTENT,
    }
    return table
