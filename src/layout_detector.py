"""Primary page-layout detector adapter: Aryn/deformable-detr-DocLayNet.

Factory pattern: `build_layout_detector(model_id)` returns a plain
`detector(image) -> list[Region]` callable that can be injected into
`detect_layout` (DESIGN_SPEC §4.1). All heavy imports (transformers, torch)
are deferred to the factory call so this module can be imported on CPU-only
machines without crashing.

Contract:
    - Input: a PIL.Image.Image (any mode; processor handles conversion).
    - Output: `list[Region]`, label normalized, source="layout", boxes in xyxy
      pixel coords for the input image size.
    - Only detections with score >= `threshold` are returned.
    - `detect_layout` may return low-score tables when they triggered the fallback
      path; callers that crop should additionally filter by score.

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


def build_layout_detector(
    model_id: str = config.LAYOUT_MODEL,
    threshold: float = 0.5,
    device: str | None = None,
) -> Callable[["Image"], list[Region]]:
    """Load the Aryn DocLayNet detector and return a detector callable.

    The model is loaded once inside this factory; the returned callable is
    stateless (model/processor captured in closure) and reusable across pages.

    Args:
        model_id: HuggingFace model id. Default: `config.LAYOUT_MODEL`.
        threshold: Confidence threshold passed to `post_process_object_detection`.
            Detections below this are filtered out at inference time.
        device: "cuda", "cpu", or None (auto-detect via `torch.cuda.is_available`).

    Returns:
        `detector(image) -> list[Region]` callable with source="layout".
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
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
            source="layout",
        )

    return detector
