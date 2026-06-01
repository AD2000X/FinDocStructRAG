"""Templated QA generation tests (CPU, synthetic) - Phase 1C.

Lock the question phrasing, gold answer = GT cell, answer_type, and the relevance judgment.
No I/O, no model.
"""

from src.qa_templates import (
    ANSWER_TYPE_NUMERIC,
    ANSWER_TYPE_TEXT,
    QA_SOURCE_TEMPLATED,
    generate_lookup_questions,
)


def _table(meta_id="ABC_2018_page_1_table_0", header_2018="2018", header_2017="2017"):
    return {
        "num_rows": 3, "num_cols": 3,
        "cells": [
            {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
             "text": "", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2,
             "text": header_2018, "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 2, "col_end": 3,
             "text": header_2017, "is_header": True},
            {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1, "text": "Revenue"},
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": "13,223"},
            {"row_start": 1, "row_end": 2, "col_start": 2, "col_end": 3, "text": "10,376"},
            {"row_start": 2, "row_end": 3, "col_start": 0, "col_end": 1, "text": "Cost"},
            {"row_start": 2, "row_end": 3, "col_start": 1, "col_end": 2, "text": "5,483"},
            {"row_start": 2, "row_end": 3, "col_start": 2, "col_end": 3, "text": "5,510"},
        ],
        "meta": {"sample_id": meta_id, "text_source": "gt"},
    }


def test_year_header_phrasing_and_gold_from_gt():
    qs = generate_lookup_questions(_table())
    assert len(qs) == 4
    first = qs[0]
    assert first["question"] == "What was Revenue in 2018?"
    assert first["gold_answer"] == "13,223"
    assert first["answer_type"] == ANSWER_TYPE_NUMERIC
    assert first["sample_id"] == "ABC_2018_page_1_table_0"
    assert first["relevant_chunk_ids"] == ["table:ABC_2018_page_1_table_0"]
    assert first["source"] == QA_SOURCE_TEMPLATED
    assert first["is_answerable"] is True
    assert "question_id" not in first  # runner assigns ids after sampling


def test_non_year_header_phrasing():
    qs = generate_lookup_questions(_table(header_2018="Amount", header_2017="Prior"))
    questions = {q["question"] for q in qs}
    assert "What was the Amount of Revenue?" in questions
    assert "What was the Prior of Cost?" in questions


def test_text_answer_type_for_non_numeric_value():
    table = {
        "num_rows": 2, "num_cols": 2,
        "cells": [
            {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
             "text": "", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2,
             "text": "Auditor", "is_header": True},
            {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1, "text": "Firm"},
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": "Deloitte"},
        ],
        "meta": {"sample_id": "X", "text_source": "gt"},
    }
    qs = generate_lookup_questions(table)
    assert len(qs) == 1
    assert qs[0]["answer_type"] == ANSWER_TYPE_TEXT
    assert qs[0]["gold_answer"] == "Deloitte"


def test_skips_rows_without_label_or_empty_values():
    table = {
        "num_rows": 3, "num_cols": 2,
        "cells": [
            {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
             "text": "", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2,
             "text": "2018", "is_header": True},
            # row 1: no label -> skipped
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": "99"},
            # row 2: label but empty value -> no question
            {"row_start": 2, "row_end": 3, "col_start": 0, "col_end": 1, "text": "Cash"},
            {"row_start": 2, "row_end": 3, "col_start": 1, "col_end": 2, "text": ""},
        ],
        "meta": {"sample_id": "X", "text_source": "gt"},
    }
    assert generate_lookup_questions(table) == []
