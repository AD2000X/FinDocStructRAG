"""Layout-region parsing + the sequential/fallback selection for Phase 2 (pure, no GPU/model).

The page detector and the table-transformer-detection fallback emit raw label strings in their
own vocabularies; this module reconciles them to one canonical lowercase vocabulary
(`normalize_label` / `LAYOUT_LABEL_MAP`, DESIGN_SPEC §9), wraps detections in a provider-neutral
`Region`, and runs the sequential-first + low-confidence-fallback selection (DESIGN_SPEC §4.1).

The model lives behind an injected `detector(image) -> list[Region]` callable (the
`llm_client.complete` pattern), so the whole page->regions path is unit-tested with fake
detectors and this module imports no transformers / torch. The real Aryn primary adapter is in
`layout_detector.py`; the table-transformer-detection fallback adapter is in `table_detection.py`.

NOTE: `detect_layout` returns candidate regions — including primary tables whose score was below
`min_table_score` (they triggered the fallback but were not removed). Callers that crop must
apply their own `score >= threshold` filter; see `layout_detector.py` and `table_detection.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from . import bbox_utils

TABLE_LABEL = "table"

# Raw detector label string -> canonical lowercase label. Explicit (not a blind `.lower()`) so the
# two detectors' table strings reconcile to one "table" and the DocLayNet classes are pinned.
LAYOUT_LABEL_MAP: dict[str, str] = {
    # Aryn/deformable-detr-DocLayNet id2label values
    "N/A": "n/a",
    "Caption": "caption",
    "Footnote": "footnote",
    "Formula": "formula",
    "List-item": "list-item",
    "Page-footer": "page-footer",
    "Page-header": "page-header",
    "Picture": "picture",
    "Section-header": "section-header",
    "Table": "table",
    "Text": "text",
    "Title": "title",
    # microsoft/table-transformer-detection labels (the fallback)
    "table": "table",
    "table rotated": "table",
}


def normalize_label(raw: str) -> str:
    """Raw detector label -> canonical lowercase label.

    Known labels go through `LAYOUT_LABEL_MAP`; an unknown label degrades to a lowercase slug
    (spaces -> hyphens) so it stays deterministic and comparable rather than crashing.
    """
    if raw in LAYOUT_LABEL_MAP:
        return LAYOUT_LABEL_MAP[raw]
    return raw.strip().lower().replace(" ", "-")


@dataclass(frozen=True)
class Region:
    """A detected layout region (provider-neutral; what the eval and crop steps consume).

    label: canonical (normalize_label'd) class. score: confidence. box: xyxy pixel coords.
    source: which detector produced it ("layout" | "table_fallback"). Labels are expected already
    normalized (the detector adapter calls `normalize_label`); box and score are coerced here.
    """

    label: str
    score: float
    box: tuple[float, float, float, float]
    source: str

    def __post_init__(self) -> None:
        box = tuple(float(c) for c in self.box)
        if len(box) != 4:
            raise ValueError(f"Region.box must have 4 coords, got {len(box)}: {self.box}")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "score", float(self.score))


def detections_to_regions(
    scores: Sequence[float],
    labels: Sequence[int],
    boxes: Sequence[Sequence[float]],
    id2label: dict[int, str],
    *,
    source: str,
) -> list[Region]:
    """Convert raw detector output (post-processed lists) to a list of canonical Regions.

    The detector adapter calls this after `post_process_object_detection`; the result is what
    `detect_layout` / the rest of Phase 2 consumes. No torch here - inputs are plain lists/dicts
    so the shared conversion is unit-tested locally without a model.

    Fail-fast on length mismatch and on unknown label ids (a new detector with different classes
    should be noticed immediately, not silently dropped).
    """
    if not (len(scores) == len(labels) == len(boxes)):
        raise ValueError(
            f"detections_to_regions: length mismatch - "
            f"scores={len(scores)}, labels={len(labels)}, boxes={len(boxes)}"
        )
    regions = []
    for score, label_id, box in zip(scores, labels, boxes):
        if label_id not in id2label:
            raise KeyError(
                f"detections_to_regions: unknown label id {label_id!r} not in id2label "
                f"(known ids: {sorted(id2label)})"
            )
        regions.append(Region(
            label=normalize_label(id2label[label_id]),
            score=float(score),
            box=tuple(float(c) for c in box),  # type: ignore[arg-type]
            source=source,
        ))
    return regions


def detect_layout(
    image,
    detector: Callable[[object], Sequence[Region]],
    fallback_detector: Callable[[object], Sequence[Region]] | None = None,
    *,
    min_table_score: float = 0.5,
    dedup_iou: float = 0.5,
) -> list[Region]:
    """Sequential-first layout detection with a low-confidence table fallback (DESIGN_SPEC §4.1).

    Run the primary detector. If it produced no table region scoring >= `min_table_score`, run the
    fallback detector and merge in ITS table regions only (the fallback has no non-table class
    semantics). Overlapping table regions are deduped by IoU - highest score wins, deterministic
    tie-break - and dedup is applied ONLY among table regions, so a fallback table never suppresses
    a primary text/figure and vice versa. Returns every region (all classes), table-deduped, so
    AP/IoU can use the full layout and the crop step just filters `label == "table"`.

    `detector` / `fallback_detector` are injected callables returning already-normalized Regions,
    so this stays pure and testable with fakes (no model here).
    """
    regions = list(detector(image))
    strong_tables = [r for r in regions if r.label == TABLE_LABEL and r.score >= min_table_score]
    if not strong_tables and fallback_detector is not None:
        regions = regions + [r for r in fallback_detector(image) if r.label == TABLE_LABEL]

    table_idx = [i for i, r in enumerate(regions) if r.label == TABLE_LABEL]
    if len(table_idx) > 1:
        boxes = [regions[i].box for i in table_idx]
        scores = [regions[i].score for i in table_idx]
        keep_orig = {table_idx[j] for j in bbox_utils.dedup_boxes(boxes, scores, dedup_iou)}
        regions = [r for i, r in enumerate(regions) if r.label != TABLE_LABEL or i in keep_orig]
    return regions
