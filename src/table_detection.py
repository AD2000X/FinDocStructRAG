"""Table-detection fallback adapter: microsoft/table-transformer-detection.

Factory pattern: `build_table_transformer_detector(model_id, threshold)` returns
a plain `detector(image) -> list[Region]` callable. Used as the `fallback_detector`
argument to `detect_layout` when the primary layout detector finds no confident table.

The fallback emits only "table" / "table rotated" classes (both normalize to "table"),
which is exactly the contract `detect_layout` expects: fallback regions that are not
"table" are discarded by `detect_layout` before merging. Source is "table_fallback".

Contract:
    - Input: a PIL.Image.Image.
    - Output: `list[Region]`, source="table_fallback", boxes in xyxy pixel coords.
    - Only detections with score >= `threshold` are returned.
    - Callers that crop should additionally filter by score (same note as
      `layout_detector.py`).

NOT tested in ordinary pytest (requires the model). Exercise via
`scripts/smoke_layout_detector.py` or the Phase 2 Colab notebook.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from . import config
from .layout_parsing import Region, detections_to_regions

if TYPE_CHECKING:
    from PIL.Image import Image


def build_table_transformer_detector(
    model_id: str = config.TATR_DETECTION_MODEL,
    threshold: float = 0.5,
    device: str | None = None,
) -> Callable[["Image"], list[Region]]:
    """Load the table-transformer-detection model and return a detector callable.

    The model is loaded once inside this factory; the returned callable is
    stateless (model/processor captured in closure) and reusable across pages.

    Args:
        model_id: HuggingFace model id. Default: `config.TATR_DETECTION_MODEL`.
        threshold: Confidence threshold. Detections below this are filtered out.
        device: "cuda", "cpu", or None (auto-detect).

    Returns:
        `detector(image) -> list[Region]` callable with source="table_fallback".
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id)
    model = model.to(device).eval()
    id2label: dict[int, str] = model.config.id2label

    def detector(image: "Image") -> list[Region]:
        w, h = image.size
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[h, w]])
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        return detections_to_regions(
            scores=results["scores"].tolist(),
            labels=results["labels"].tolist(),
            boxes=results["boxes"].tolist(),
            id2label=id2label,
            source="table_fallback",
        )

    return detector
