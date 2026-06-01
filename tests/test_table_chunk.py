"""Table chunk builder tests (CPU, synthetic) - Phase 1C.

Lock the chunk schema and the corpus-independent chunk_id. No I/O.
"""

from src.table_chunk import build_chunk, chunk_id_for
from src.table_serialize import SERIALIZE_LINEARIZED, SERIALIZE_MARKDOWN


def _table(text_source):
    return {
        "num_rows": 2, "num_cols": 2,
        "cells": [
            {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
             "text": "", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2,
             "text": "2018", "is_header": True},
            {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1, "text": "Revenue"},
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": "13,223"},
        ],
        "meta": {"sample_id": "ABC_2018_page_1_table_0", "text_source": text_source,
                 "evaluation_type": "content"},
    }


def test_build_chunk_carries_provenance_and_linearized_text():
    chunk = build_chunk(_table("ocr"), serialization=SERIALIZE_LINEARIZED)
    assert chunk["chunk_id"] == "table:ABC_2018_page_1_table_0"
    assert chunk["sample_id"] == "ABC_2018_page_1_table_0"
    assert chunk["text_source"] == "ocr"
    assert chunk["serialization"] == "linearized"
    assert chunk["text"] == "Revenue: 2018 = 13,223"
    assert chunk["num_rows"] == 2 and chunk["num_cols"] == 2


def test_chunk_id_is_corpus_independent():
    # Same table, different source/serialization -> same chunk_id (the relevance judgment).
    gt = build_chunk(_table("gt"), serialization=SERIALIZE_MARKDOWN)
    ocr = build_chunk(_table("ocr"), serialization=SERIALIZE_LINEARIZED)
    assert gt["chunk_id"] == ocr["chunk_id"] == chunk_id_for("ABC_2018_page_1_table_0")
    assert gt["text_source"] != ocr["text_source"]


def test_build_chunk_markdown_mode():
    chunk = build_chunk(_table("gt"), serialization=SERIALIZE_MARKDOWN)
    assert chunk["serialization"] == "markdown"
    assert chunk["text"].splitlines()[0] == "|  | 2018 |"


def test_build_chunk_defaults_unknown_meta():
    chunk = build_chunk({"num_rows": 0, "num_cols": 0, "cells": []})
    assert chunk["chunk_id"] == "table:unknown"
    assert chunk["text_source"] == "none"
    assert chunk["text"] == ""
