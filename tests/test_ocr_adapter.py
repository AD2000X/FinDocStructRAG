"""OCR adapter parsing tests (CPU, synthetic).

Lock the PaddleOCR-3.x result -> OCRWord mapping without importing paddle. The page
dicts mirror the real predict() output captured on Colab (return_word_box=True): a
line carries word-level tokens in `text_word` and a parallel per-token polygon list in
`text_word_region`, both including whitespace tokens; the only score is per line.
"""

from src.ocr_adapter import _parse_v3


def test_parse_v3_prefers_word_level_and_drops_whitespace():
    # "External Sales" on one line, "$ 45,854" on the next. Whitespace tokens (" ")
    # carry their own quad and must be dropped, not emitted as empty words.
    page = {
        "rec_texts": ["External Sales", "$ 45,854"],
        "rec_scores": [0.99, 0.97],
        "text_word": [
            ["External", " ", "Sales"],
            ["$", " ", "45,854"],
        ],
        "text_word_region": [
            [((34, 0), (81, 0), (81, 15), (34, 15)),
             ((81, 0), (84, 0), (84, 15), (81, 15)),
             ((89, 0), (116, 0), (116, 15), (89, 15))],
            [((10, 20), (18, 20), (18, 35), (10, 35)),
             ((18, 20), (22, 20), (22, 35), (18, 35)),
             ((30, 20), (90, 20), (90, 35), (30, 35))],
        ],
    }
    words = _parse_v3([page])

    assert [w.text for w in words] == ["External", "Sales", "$", "45,854"]
    # axis-aligned enclosing box of the token's quad
    assert words[0].bbox == [34.0, 0.0, 81.0, 15.0]
    assert words[3].bbox == [30.0, 20.0, 90.0, 35.0]
    # every word on a line shares that line's score
    assert words[0].confidence == 0.99
    assert words[3].confidence == 0.97
    assert all(w.source == "paddleocr" for w in words)


def test_parse_v3_falls_back_to_line_level_without_word_boxes():
    # An older build / flag off: no text_word, only line-level rec_polys.
    page = {
        "rec_texts": ["Total"],
        "rec_scores": [0.95],
        "rec_polys": [[[0, 0], [50, 0], [50, 12], [0, 12]]],
    }
    words = _parse_v3([page])
    assert len(words) == 1
    assert words[0].text == "Total"
    assert words[0].bbox == [0.0, 0.0, 50.0, 12.0]
    assert words[0].confidence == 0.95


def test_parse_v3_empty_result():
    assert _parse_v3([]) == []
    assert _parse_v3(None) == []
