"""Templated QA generation from GT-filled tables (Phase 1C).

Lookup / numeric questions generated straight from GT cells: the question is built from a
row label + column header, and the answer IS the GT cell value. Because the answer comes
from ground truth, both the gold answer and the relevance judgment (the source table's
chunk) are automatic - no manual transcription, no hallucination. The harder reasoning /
comparison / unanswerable questions cannot be templated and are hand-authored separately
(config.QA_MANUAL_SEED).

Known proxy limitation: a templated question ("What was Revenue in 2018?") can be answerable
from more than one table when issuers share row/column labels, so its single relevant_chunk
judgment understates retrieval precision. Identifiers are deliberately NOT leaked into the
question (that would make BM25 retrieval trivial); the hand-authored set adds disambiguated
questions instead. Reported as a limitation, not silently tuned away.
"""

from __future__ import annotations

import re

from .numeric_utils import looks_numeric
from .table_chunk import chunk_id_for
from .table_serialize import table_grid

QA_SOURCE_TEMPLATED = "templated_gt"
ANSWER_TYPE_NUMERIC = "numeric"
ANSWER_TYPE_TEXT = "text"

_YEAR = re.compile(r"(?:19|20)\d{2}")


def _text(cell) -> str:
    return (cell.get("text", "") or "").strip() if cell else ""


def _question(row_label: str, col_header: str) -> str:
    """Phrasing adapts to the column header: a year/period reads as 'in <year>'."""
    if _YEAR.fullmatch(col_header):
        return f"What was {row_label} in {col_header}?"
    return f"What was the {col_header} of {row_label}?"


def generate_lookup_questions(table) -> list[dict]:
    """One lookup QA record per GT body cell with a non-empty row label, header, and value.

    Records carry no question_id (the runner assigns ids once it has sampled across tables).
    """
    grid, n_rows, n_cols, header_rows, col_headers = table_grid(table)
    header_set = set(header_rows)
    sample_id = table.get("meta", {}).get("sample_id", "unknown")
    cid = chunk_id_for(sample_id)

    records = []
    for r in range(n_rows):
        if r in header_set:
            continue
        row_label = _text(grid[r][0])
        if not row_label:
            continue
        for c in range(1, n_cols):
            value = _text(grid[r][c])
            col_header = col_headers[c]
            if not value or not col_header:
                continue
            records.append({
                "question": _question(row_label, col_header),
                "gold_answer": value,
                "answer_type": (ANSWER_TYPE_NUMERIC if looks_numeric(value)
                                else ANSWER_TYPE_TEXT),
                "sample_id": sample_id,
                "relevant_chunk_ids": [cid],
                "source": QA_SOURCE_TEMPLATED,
                "is_answerable": True,
            })
    return records
