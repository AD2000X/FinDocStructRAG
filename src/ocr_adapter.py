"""OCR adapter (DESIGN_SPEC §5.5, Phase 1B).

Normalises an OCR engine's output into a flat list of OCRWord records that
assign_words_to_cells() can consume. PaddleOCR is the only engine wired in for now
(PLAN: PaddleOCR priority over Tesseract); a Tesseract fallback can slot in behind the
same OCRWord contract later.

PaddleOCR is a GPU/Colab-only dependency, so it is imported lazily inside run_paddleocr.
Importing this module (and the OCRWord dataclass) stays pure-CPU and local-test-safe.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCRWord:
    """One recognised word, in the crop's pixel coordinates.

    bbox is axis-aligned [x1, y1, x2, y2]; engines that return a quadrilateral are
    reduced to its enclosing box. source names the engine so mixed runs stay traceable.
    """

    text: str
    bbox: list[float]
    confidence: float
    source: str

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "source": self.source,
        }


def _quad_to_bbox(quad) -> list[float]:
    """Enclosing axis-aligned box of a 4-point polygon (PaddleOCR's box format).

    Casts to float: PaddleOCR returns numpy coordinates, which are not JSON-serialisable.
    """
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def build_paddleocr():
    """Build a reusable PaddleOCR instance (English).

    Construct once per batch and pass into run_paddleocr; building per sample reloads
    the model weights every time. Imported lazily so this module stays CPU/local-safe.
    Only `lang` is passed: PaddleOCR 3.x dropped `show_log` / `use_angle_cls`, so keeping
    the constructor minimal avoids version-specific "Unknown argument" errors.

    oneDNN must be disabled or the paddlepaddle 3.x CPU build crashes in the oneDNN path
    of the inference predictor ("ConvertPirAttribute2RuntimeAttribute not support"). The
    effective switch is the predictor's enable_mkldnn=False (the global FLAGS_* do not
    reach the paddlex inference predictor); the FLAGS are still set as a harmless belt-
    and-braces before paddle imports. enable_mkldnn is passed in a try/except because old
    PaddleOCR builds reject unknown constructor args.

    return_word_box=True makes the engine emit word-level tokens and per-word boxes
    (text_word / text_word_region) in addition to the line-level result. Financial tables
    pack several narrow numeric columns close together, and the line-level detector merges
    them into one box that straddles column boundaries; word boxes keep each token in its
    own column so assign_words_to_cells can place them correctly.
    """
    import os

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    from paddleocr import PaddleOCR

    # Table crops are already cropped, upright and flat, so the doc-orientation,
    # unwarping and textline-orientation sub-models are wasted CPU (and downloads). Keep
    # only detection + recognition. Guarded: older builds that reject these args fall
    # back to a plain instance.
    try:
        return PaddleOCR(
            lang="en",
            enable_mkldnn=False,
            return_word_box=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except (TypeError, ValueError):
        return PaddleOCR(lang="en")


def _parse_v3(result) -> list[OCRWord]:
    """PaddleOCR 3.x: predict() returns dict-like results with parallel lists.

    With return_word_box=True each line carries word-level tokens in `text_word` and a
    parallel per-token polygon list in `text_word_region` (both include whitespace tokens
    like " ", which are dropped). These are preferred so adjacent financial columns are
    not merged into one line-level box. The only recognition score is per line
    (rec_scores), so every word on a line shares its line's score. When the word-level
    keys are absent (older build, or the flag was off) this falls back to the line-level
    rec_texts / rec_polys.
    """
    words: list[OCRWord] = []
    for page in result or []:
        scores = page.get("rec_scores") or []
        text_word = page.get("text_word")
        regions = page.get("text_word_region")
        if text_word and regions:
            for line_idx, (tokens, quads) in enumerate(zip(text_word, regions)):
                score = float(scores[line_idx]) if line_idx < len(scores) else 1.0
                for token, quad in zip(tokens, quads):
                    if not token.strip():
                        continue
                    words.append(OCRWord(
                        text=token,
                        bbox=_quad_to_bbox(quad),
                        confidence=score,
                        source="paddleocr",
                    ))
            continue
        texts = page["rec_texts"]
        polys = page.get("rec_polys")
        if polys is None:
            polys = page.get("dt_polys")
        for text, score, poly in zip(texts, scores, polys):
            words.append(OCRWord(
                text=text,
                bbox=_quad_to_bbox(poly),
                confidence=float(score),
                source="paddleocr",
            ))
    return words


def _parse_v2(result) -> list[OCRWord]:
    """PaddleOCR 2.x: ocr() returns [ [ [quad, (text, conf)], ... ] ]."""
    words: list[OCRWord] = []
    for page in result or []:
        for quad, (text, conf) in page or []:
            words.append(OCRWord(
                text=text,
                bbox=_quad_to_bbox(quad),
                confidence=float(conf),
                source="paddleocr",
            ))
    return words


def run_paddleocr(image, ocr=None) -> list[OCRWord]:
    """Run PaddleOCR on a crop and return normalised OCRWord records.

    image: a PIL.Image or numpy array of the table crop.
    ocr: a pre-built PaddleOCR instance (reused across samples on Colab); built here if
    None. PaddleOCR is imported lazily so this module imports without the GPU stack.
    Handles both the 3.x predict() API and the 2.x ocr() API.
    """
    import numpy as np

    if ocr is None:
        ocr = build_paddleocr()

    arr = np.asarray(image)
    if hasattr(ocr, "predict"):
        return _parse_v3(ocr.predict(arr))
    return _parse_v2(ocr.ocr(arr, cls=False))
