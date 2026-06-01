"""Word-to-cell assignment tests (CPU, synthetic) - DESIGN_SPEC 5.5, Phase 1B.

Lock the placement rules (center-in-cell, IoU fallback, reading-order sort, unassigned
logging) with hand-built cells and words. No OCR engine, no image.
"""

from src.tatr_postprocess import assign_words_to_cells, join_word_tokens
from src.ocr_adapter import OCRWord


class _RecordingLogger:
    """Captures FailureLogger.log calls without touching disk."""

    def __init__(self):
        self.calls = []

    def log(self, sample_id, phase, error_type="unknown", message=""):
        self.calls.append((sample_id, phase, error_type, message))


def _word(text, bbox):
    return {"text": text, "bbox": bbox, "confidence": 0.99, "source": "synthetic"}


def _two_row_grid():
    # one column, two stacked rows.
    return [
        {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
         "bbox": [0, 0, 100, 50], "text": "", "is_header": False, "words": []},
        {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1,
         "bbox": [0, 50, 100, 100], "text": "", "is_header": False, "words": []},
    ]


def test_center_in_cell_assignment():
    cells = _two_row_grid()
    assign_words_to_cells(
        cells, [_word("top", [10, 10, 40, 40]), _word("bottom", [10, 60, 40, 90])])
    assert cells[0]["text"] == "top"
    assert cells[1]["text"] == "bottom"


def test_words_sorted_reading_order():
    cells = [{"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
              "bbox": [0, 0, 200, 100], "text": "", "is_header": False, "words": []}]
    # supplied out of order; lower y must come first, then lower x.
    assign_words_to_cells(cells, [
        _word("B", [100, 40, 140, 60]),
        _word("A", [10, 5, 40, 25]),
    ])
    assert cells[0]["text"] == "A B"


def test_iou_fallback_when_center_outside():
    cells = [{"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
              "bbox": [0, 0, 100, 100], "text": "", "is_header": False, "words": []}]
    # center (125,125) is outside the cell, but the box overlaps the cell corner.
    assign_words_to_cells(cells, [_word("edge", [90, 90, 160, 160])])
    assert cells[0]["text"] == "edge"


def test_unassigned_word_is_logged_not_placed():
    cells = [{"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
              "bbox": [0, 0, 100, 100], "text": "", "is_header": False, "words": []}]
    log = _RecordingLogger()
    assign_words_to_cells(
        cells, [_word("stray", [200, 200, 250, 250])], logger=log, sample_id="s1")
    assert cells[0]["text"] == ""
    assert len(log.calls) == 1
    sample_id, _phase, error_type, message = log.calls[0]
    assert sample_id == "s1"
    assert error_type == "word_assignment"
    assert message == "stray"


def _grid_2x2():
    # rows 20..120 / 120..220, cols 0..50 / 50..100 (a tall grid so the 5% y-margin
    # leaves room above row 0 for a header word).
    return [
        {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
         "bbox": [0, 20, 50, 120], "text": "", "is_header": False, "words": []},
        {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2,
         "bbox": [50, 20, 100, 120], "text": "", "is_header": False, "words": []},
        {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1,
         "bbox": [0, 120, 50, 220], "text": "", "is_header": False, "words": []},
        {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2,
         "bbox": [50, 120, 100, 220], "text": "", "is_header": False, "words": []},
    ]


def test_assign_word_above_first_row_goes_to_nearest_row_col():
    cells = _grid_2x2()
    # center (75, 15): above row 0 (starts y=20) but inside the expanded grid bbox;
    # x falls in column 1. Should snap to cell (row 0, col 1).
    assign_words_to_cells(cells, [_word("H", [55, 12, 95, 18])])
    placed = {(c["row_start"], c["col_start"]): c["text"] for c in cells}
    assert placed[(0, 1)] == "H"
    assert all(t == "" for k, t in placed.items() if k != (0, 1))


def test_far_outside_word_remains_unassigned():
    cells = _grid_2x2()
    log = _RecordingLogger()
    # center y = -95, far above the expanded grid bbox -> not snapped.
    assign_words_to_cells(
        cells, [_word("footer", [55, -100, 95, -90])], logger=log, sample_id="s")
    assert all(c["text"] == "" for c in cells)
    assert len(log.calls) == 1
    assert log.calls[0][2] == "word_assignment"


def test_center_in_cell_still_wins():
    cells = _grid_2x2()
    # center (25, 70) sits squarely inside cell (0,0); must not be moved by fallback.
    assign_words_to_cells(cells, [_word("x", [10, 60, 40, 80])])
    placed = {(c["row_start"], c["col_start"]): c["text"] for c in cells}
    assert placed[(0, 0)] == "x"


def test_iou_fallback_still_works():
    cells = [{"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
              "bbox": [0, 0, 100, 100], "text": "", "is_header": False, "words": []}]
    # center (125,125) outside, but the box overlaps the cell corner -> IoU assigns.
    assign_words_to_cells(cells, [_word("edge", [90, 90, 160, 160])])
    assert cells[0]["text"] == "edge"


def test_ocrword_to_dict():
    w = OCRWord(text="x", bbox=[1, 2, 3, 4], confidence=0.5, source="paddleocr")
    assert w.to_dict() == {
        "text": "x", "bbox": [1, 2, 3, 4], "confidence": 0.5, "source": "paddleocr"}


# --- clean token join (word-level OCR de-spacing) --------------------------

def test_join_currency_number_and_parens():
    assert join_word_tokens(["$", "13", ",", "223"]) == "$13,223"
    assert join_word_tokens(["$", "(", "250", ",", "721", ")"]) == "$(250,721)"
    assert join_word_tokens(["(", "Unaudited", ")"]) == "(Unaudited)"


def test_join_percent_and_separators():
    assert join_word_tokens(["7.50", "%"]) == "7.50%"
    assert join_word_tokens(["13", ",", "223"]) == "13,223"
    # a list comma between words keeps its trailing space
    assert join_word_tokens(["Expenses", ",", "Net", "of"]) == "Expenses, Net of"


def test_join_apostrophe_contraction_vs_possessive():
    # short suffix contracts; a full following word keeps its space
    assert join_word_tokens(["Management", "'", "s"]) == "Management's"
    assert join_word_tokens(["Stockholders", "'", "Equity"]) == "Stockholders' Equity"


def test_join_does_not_rewrite_real_misread():
    # comma OCR'd as a period stays a period (a genuine error, not whitewashed)
    assert join_word_tokens(["29", ".", "2018"]) == "29.2018"
    # plain words keep normal spacing
    assert join_word_tokens(["External", "Sales", "By", "Major"]) == "External Sales By Major"
