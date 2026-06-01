"""Table serialization tests (CPU, synthetic) - Phase 1C.

Lock the markdown vs linearized renderings, including header detection and spanning
headers. No model, no I/O.
"""

import pytest

from src.table_serialize import (
    SERIALIZE_LINEARIZED,
    SERIALIZE_MARKDOWN,
    serialize,
    serialize_linearized,
    serialize_markdown,
)


def _financial_3x3():
    # header row 0: ["", "2018", "2017"]; two body rows with a label in col 0.
    return {
        "num_rows": 3,
        "num_cols": 3,
        "cells": [
            {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
             "text": "", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2,
             "text": "2018", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 2, "col_end": 3,
             "text": "2017", "is_header": True},
            {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1, "text": "Revenue"},
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": "13,223"},
            {"row_start": 1, "row_end": 2, "col_start": 2, "col_end": 3, "text": "10,376"},
            {"row_start": 2, "row_end": 3, "col_start": 0, "col_end": 1, "text": "Cost"},
            {"row_start": 2, "row_end": 3, "col_start": 1, "col_end": 2, "text": "5,483"},
            {"row_start": 2, "row_end": 3, "col_start": 2, "col_end": 3, "text": "5,510"},
        ],
    }


def test_markdown_grid_with_header_separator():
    out = serialize_markdown(_financial_3x3())
    assert out.splitlines() == [
        "|  | 2018 | 2017 |",
        "| --- | --- | --- |",
        "| Revenue | 13,223 | 10,376 |",
        "| Cost | 5,483 | 5,510 |",
    ]


def test_linearized_pairs_value_with_column_header():
    out = serialize_linearized(_financial_3x3())
    assert out.splitlines() == [
        "Revenue: 2018 = 13,223; 2017 = 10,376",
        "Cost: 2018 = 5,483; 2017 = 5,510",
    ]


def test_linearized_skips_empty_values():
    table = {
        "num_rows": 2, "num_cols": 3,
        "cells": [
            {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
             "text": "", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2,
             "text": "2018", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 2, "col_end": 3,
             "text": "2017", "is_header": True},
            {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1, "text": "Revenue"},
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": ""},
            {"row_start": 1, "row_end": 2, "col_start": 2, "col_end": 3, "text": "10,376"},
        ],
    }
    assert serialize_linearized(table) == "Revenue: 2017 = 10,376"


def test_linearized_spanning_header_applies_to_each_column():
    # row 0: "Year" spans cols 1-2 (header); row 1: sub-headers 2018 / 2017 (header).
    table = {
        "num_rows": 3, "num_cols": 3,
        "cells": [
            {"row_start": 0, "row_end": 2, "col_start": 0, "col_end": 1,
             "text": "Item", "is_header": True},
            {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 3,
             "text": "Year", "is_header": True},
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2,
             "text": "2018", "is_header": True},
            {"row_start": 1, "row_end": 2, "col_start": 2, "col_end": 3,
             "text": "2017", "is_header": True},
            {"row_start": 2, "row_end": 3, "col_start": 0, "col_end": 1, "text": "Revenue"},
            {"row_start": 2, "row_end": 3, "col_start": 1, "col_end": 2, "text": "13,223"},
            {"row_start": 2, "row_end": 3, "col_start": 2, "col_end": 3, "text": "10,376"},
        ],
    }
    # "Year" (spanned over both cols) then the per-column sub-header.
    assert serialize_linearized(table) == (
        "Revenue: Year 2018 = 13,223; Year 2017 = 10,376"
    )


def test_markdown_escapes_pipe():
    table = {"num_rows": 1, "num_cols": 1, "cells": [
        {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1, "text": "a|b"}]}
    assert serialize_markdown(table).splitlines()[0] == r"| a\|b |"


def test_serialize_dispatch_and_unknown_mode():
    table = _financial_3x3()
    assert serialize(table, SERIALIZE_MARKDOWN) == serialize_markdown(table)
    assert serialize(table, SERIALIZE_LINEARIZED) == serialize_linearized(table)
    with pytest.raises(ValueError):
        serialize(table, "bogus")


def test_empty_table_serializes_to_empty_string():
    assert serialize_markdown({"num_rows": 0, "num_cols": 0, "cells": []}) == ""
    assert serialize_linearized({"num_rows": 0, "num_cols": 0, "cells": []}) == ""
