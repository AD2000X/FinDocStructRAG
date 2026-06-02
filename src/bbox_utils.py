"""Geometry helpers for Phase 2 layout boxes (pure, no GPU, no model).

The layout detector returns boxes as xyxy in page-pixel coordinates (see
`scripts/smoke_layout_detector.py`); DocLayNet / COCO GT is xywh. These are the small,
unit-tested primitives the rest of Phase 2 (`layout_parsing`, `table_detection`) builds on:
format conversion, clamping, IoU, score-ordered dedup, and a safe crop.

Boxes are plain `(x1, y1, x2, y2)` float tuples/lists - `bbox_utils` is the low-level geometry
layer and carries no label/score/source (that is `layout_parsing.Region`, later). Geometry
stays in float; only the final crop snaps to integer pixels.

Normalized (0-1) <-> pixel and render-DPI conversions are deliberately omitted: the contract is
pixel xyxy end to end (detector output and DocLayNet page images), so a normalized path would
be speculative. Add a small converter if a PDF renderer or normalized GT ever appears.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Box = Sequence[float]  # (x1, y1, x2, y2)


def xyxy_to_xywh(box: Box) -> tuple[float, float, float, float]:
    """(x1, y1, x2, y2) -> COCO (x, y, w, h)."""
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_xyxy(box: Box) -> tuple[float, float, float, float]:
    """COCO (x, y, w, h) -> (x1, y1, x2, y2)."""
    x, y, w, h = box
    return (x, y, x + w, y + h)


def clamp_box(box: Box, width: float, height: float) -> tuple[float, float, float, float]:
    """Clamp an xyxy box to the page [0, width] x [0, height], per coordinate, staying float.

    Each coordinate is clamped independently, so a box entirely off the page collapses to a
    zero-extent box rather than erroring; the caller checks validity (e.g. via `box_area`).
    Detector boxes can run negative or past the page edge (seen in the smoke), so this is the
    primitive that normalizes them before dedup / crop.
    """
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
        min(max(x2, 0.0), width),
        min(max(y2, 0.0), height),
    )


def box_area(box: Box) -> float:
    """Area of an xyxy box; 0 for a degenerate / inverted box (never negative)."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two xyxy boxes; 0.0 when they do not overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def dedup_boxes(boxes: Sequence[Box], scores: Sequence[float], iou_threshold: float) -> list[int]:
    """Greedy score-ordered NMS. Returns kept original indices, highest score first.

    Boxes are visited in order of (descending score, ascending original index), so ties are
    deterministic. A box is dropped when it overlaps an already-kept box by IoU >= threshold.
    Used to merge the layout-table path with the `table-transformer-detection` fallback.
    """
    if len(boxes) != len(scores):
        raise ValueError(f"boxes ({len(boxes)}) and scores ({len(scores)}) length mismatch")
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    kept: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[k]) < iou_threshold for k in kept):
            kept.append(i)
    return kept


def crop_with_padding(image, box: Box, pad: float = 0.0):
    """Crop a PIL image to an xyxy box: pad, snap to whole pixels, clamp to the image.

    Geometry is float up to here. For the crop the box is expanded by `pad` on every side, then
    floored at the top-left and ceiled at the bottom-right (so rounding never shaves content),
    then clamped to the image. A box that clamps to zero / negative extent raises: a degenerate
    crop means upstream selection is wrong, so it fails loud instead of cropping garbage.

    `width`/`height` come from `image.size`, the single source of truth, so they cannot disagree
    with the image being cropped.
    """
    width, height = image.size
    x1, y1, x2, y2 = box
    ix1 = max(0, math.floor(x1 - pad))
    iy1 = max(0, math.floor(y1 - pad))
    ix2 = min(width, math.ceil(x2 + pad))
    iy2 = min(height, math.ceil(y2 + pad))
    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError(
            f"degenerate crop box {(ix1, iy1, ix2, iy2)} after clamp to image {(width, height)}"
        )
    return image.crop((ix1, iy1, ix2, iy2))
