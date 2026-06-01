"""Content-fill assembly tests (CPU, synthetic) - Phase 1B.

Check that fill_table builds the grid, assigns words, and stamps provenance, and that
parse_words_json normalises FinTabNet word records into the assignment shape.
"""

from src.canonical_schema import (
    EVAL_TYPE_CONTENT, TEXT_SOURCE_GT, TEXT_SOURCE_OCR)
from src.fintabnet_loader import parse_words_json
from src.table_fill import fill_table


def _grid_2x2():
    # two rows, two cols spanning 0..100 x 0..100.
    return {
        "row_boxes": [{"bbox": [0, 0, 100, 50]}, {"bbox": [0, 50, 100, 100]}],
        "col_boxes": [{"bbox": [0, 0, 50, 100]}, {"bbox": [50, 0, 100, 100]}],
    }


def _word(text, bbox):
    return {"text": text, "bbox": bbox, "confidence": 1.0, "source": "gt"}


def test_fill_table_places_text_and_stamps_meta():
    words = [
        _word("a", [5, 5, 20, 20]),     # top-left  -> (0,0)
        _word("b", [60, 5, 80, 20]),    # top-right -> (0,1)
        _word("c", [5, 60, 20, 80]),    # bot-left  -> (1,0)
        _word("d", [60, 60, 80, 80]),   # bot-right -> (1,1)
    ]
    table = fill_table(_grid_2x2(), words, sample_id="s1", text_source=TEXT_SOURCE_GT)
    assert table["num_rows"] == 2 and table["num_cols"] == 2
    by_pos = {(c["row_start"], c["col_start"]): c["text"] for c in table["cells"]}
    assert by_pos == {(0, 0): "a", (0, 1): "b", (1, 0): "c", (1, 1): "d"}
    assert table["meta"] == {
        "sample_id": "s1", "text_source": TEXT_SOURCE_GT,
        "evaluation_type": EVAL_TYPE_CONTENT}


def test_fill_table_text_source_ocr():
    table = fill_table(_grid_2x2(), [], sample_id="s", text_source=TEXT_SOURCE_OCR)
    assert table["meta"]["text_source"] == TEXT_SOURCE_OCR
    assert all(c["text"] == "" for c in table["cells"])


def test_parse_words_json_normalises_records():
    raw = [
        {"text": "Total", "bbox": [29.45, 0.75, 52.6, 12.39],
         "block_num": 0, "line_num": 0, "span_num": 98, "flags": 0},
    ]
    out = parse_words_json(raw)
    assert out == [
        {"text": "Total", "bbox": [29.45, 0.75, 52.6, 12.39],
         "confidence": 1.0, "source": "gt"}]
