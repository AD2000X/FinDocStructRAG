"""Word-to-cell assignment tests (CPU, synthetic) - DESIGN_SPEC 5.5, Phase 1B.

Lock the placement rules (center-in-cell, IoU fallback, reading-order sort, unassigned
logging) with hand-built cells and words. No OCR engine, no image.
"""

from src.tatr_postprocess import assign_words_to_cells
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


def test_ocrword_to_dict():
    w = OCRWord(text="x", bbox=[1, 2, 3, 4], confidence=0.5, source="paddleocr")
    assert w.to_dict() == {
        "text": "x", "bbox": [1, 2, 3, 4], "confidence": 0.5, "source": "paddleocr"}
