"""Backfill-header patch tests (CPU, synthetic) - Phase 1C.

Locks the pure marking contract used by scripts/mark_ocr_filled_headers.py: cells that
overlap a column-header box (IoMin) get is_header; the rest stay False.
"""

from scripts.mark_ocr_filled_headers import mark_table_headers


def _table():
    # Two cells stacked vertically: top row (header band) and a body row below it.
    return {
        "cells": [
            {"bbox": [0, 0, 30, 10], "is_header": False},
            {"bbox": [0, 10, 30, 20], "is_header": False},
        ]
    }


def test_marks_only_header_band():
    table = _table()
    marked = mark_table_headers(table, [{"bbox": [0, 0, 30, 10]}])
    assert marked == 1
    assert table["cells"][0]["is_header"] is True
    assert table["cells"][1]["is_header"] is False


def test_no_header_boxes_marks_nothing():
    table = _table()
    assert mark_table_headers(table, []) == 0
    assert not any(c["is_header"] for c in table["cells"])


def test_idempotent():
    table = _table()
    mark_table_headers(table, [{"bbox": [0, 0, 30, 10]}])
    assert mark_table_headers(table, [{"bbox": [0, 0, 30, 10]}]) == 1
