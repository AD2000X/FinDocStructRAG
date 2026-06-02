"""Unit tests for src/layout_parsing (pure; fake detectors, no model)."""

import pytest
from dataclasses import FrozenInstanceError

from src.layout_parsing import (
    LAYOUT_LABEL_MAP,
    Region,
    detect_layout,
    detections_to_regions,
    normalize_label,
)


def R(label, score, box, source="layout"):
    return Region(label=label, score=score, box=box, source=source)


# Minimal id2label that mirrors the Aryn detector contract (verified in smoke)
_ARYN_ID2LABEL = {
    0: "N/A", 1: "Caption", 2: "Footnote", 3: "Formula",
    4: "List-item", 5: "Page-footer", 6: "Page-header", 7: "Picture",
    8: "Section-header", 9: "Table", 10: "Text", 11: "Title",
}


# --- detections_to_regions ---


def test_detections_to_regions_basic():
    regions = detections_to_regions(
        scores=[0.9, 0.7],
        labels=[9, 10],
        boxes=[[1, 2, 3, 4], [5, 6, 7, 8]],
        id2label=_ARYN_ID2LABEL,
        source="layout",
    )
    assert len(regions) == 2
    assert regions[0].label == "table"
    assert regions[0].score == 0.9
    assert regions[0].box == (1.0, 2.0, 3.0, 4.0)
    assert regions[0].source == "layout"
    assert regions[1].label == "text"


def test_detections_to_regions_coerces_to_float():
    regions = detections_to_regions(
        scores=[1], labels=[9], boxes=[[10, 20, 30, 40]],
        id2label=_ARYN_ID2LABEL, source="layout",
    )
    assert all(isinstance(c, float) for c in regions[0].box)
    assert isinstance(regions[0].score, float)


def test_detections_to_regions_unknown_id_fails_fast():
    with pytest.raises(KeyError, match="unknown label id"):
        detections_to_regions(
            scores=[0.9], labels=[99], boxes=[[0, 0, 1, 1]],
            id2label=_ARYN_ID2LABEL, source="layout",
        )


def test_detections_to_regions_length_mismatch_fails_fast():
    with pytest.raises(ValueError, match="length mismatch"):
        detections_to_regions(
            scores=[0.9, 0.8], labels=[9], boxes=[[0, 0, 1, 1]],
            id2label=_ARYN_ID2LABEL, source="layout",
        )


def test_detections_to_regions_source_preserved():
    regions = detections_to_regions(
        scores=[0.8], labels=[9], boxes=[[0, 0, 1, 1]],
        id2label=_ARYN_ID2LABEL, source="table_fallback",
    )
    assert regions[0].source == "table_fallback"


def test_detections_to_regions_normalizes_label_via_map():
    # id2label[9] == "Table" -> normalize_label -> "table"
    regions = detections_to_regions(
        scores=[0.85], labels=[9], boxes=[[0, 0, 1, 1]],
        id2label=_ARYN_ID2LABEL, source="layout",
    )
    assert regions[0].label == "table"


def test_detections_to_regions_empty_input_returns_empty():
    assert detections_to_regions([], [], [], id2label=_ARYN_ID2LABEL, source="layout") == []


# --- normalize_label / LAYOUT_LABEL_MAP ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Table", "table"),
        ("table", "table"),  # table-transformer-detection label
        ("table rotated", "table"),
        ("Text", "text"),
        ("Title", "title"),
        ("Caption", "caption"),
        ("Footnote", "footnote"),
        ("Formula", "formula"),
        ("List-item", "list-item"),
        ("Page-header", "page-header"),
        ("Page-footer", "page-footer"),
        ("Section-header", "section-header"),
        ("Picture", "picture"),
        ("N/A", "n/a"),
    ],
)
def test_normalize_known_labels(raw, expected):
    assert normalize_label(raw) == expected


def test_normalize_unknown_degrades_to_slug():
    assert normalize_label("Weird Thing") == "weird-thing"


def test_label_map_reconciles_both_detectors_to_table():
    assert LAYOUT_LABEL_MAP["Table"] == "table"  # Aryn
    assert LAYOUT_LABEL_MAP["table"] == "table"  # table-transformer


# --- Region ---


def test_region_coerces_box_and_score_to_float():
    r = R("table", 1, (1, 2, 3, 4))
    assert r.box == (1.0, 2.0, 3.0, 4.0)
    assert all(isinstance(c, float) for c in r.box)
    assert isinstance(r.score, float) and r.score == 1.0


def test_region_is_frozen():
    r = R("table", 0.9, (0, 0, 1, 1))
    with pytest.raises(FrozenInstanceError):
        r.label = "text"


def test_region_bad_box_length_raises():
    with pytest.raises(ValueError):
        R("table", 0.9, (0, 0, 1))


# --- detect_layout: fallback trigger ---


def test_no_fallback_returns_primary_unchanged():
    regions = [R("text", 0.9, (0, 0, 10, 10)), R("title", 0.8, (0, 0, 5, 5))]
    assert detect_layout(None, lambda img: regions) == regions


def test_high_score_table_skips_fallback():
    primary = [R("table", 0.95, (0, 0, 10, 10))]

    def fallback(img):
        raise AssertionError("fallback must not run when a strong table exists")

    out = detect_layout(None, lambda img: primary, fallback, min_table_score=0.5)
    assert [r.label for r in out] == ["table"]


def test_primary_finds_no_table_skips_fallback():
    # Primary detected zero tables → fallback must NOT fire (TATR has high FP rate on table-free pages)
    primary = [R("text", 0.9, (0, 0, 10, 10))]

    def fallback(img):
        raise AssertionError("fallback must not run when primary found zero tables")

    out = detect_layout(None, lambda img: primary, fallback, min_table_score=0.5)
    assert [r.label for r in out] == ["text"]


def test_low_score_table_present_triggers_fallback():
    # Primary found a table but scored it below threshold → fallback SHOULD fire
    primary = [R("text", 0.9, (0, 0, 10, 10)), R("table", 0.3, (1, 1, 9, 9))]
    fb = [R("table", 0.8, (1, 1, 9, 9), source="table_fallback")]
    out = detect_layout(None, lambda img: primary, lambda img: fb, min_table_score=0.5)
    tables = [r for r in out if r.label == "table"]
    assert len(tables) == 1  # primary table deduped by higher-score fallback
    assert tables[0].source == "table_fallback"


def test_low_score_table_triggers_fallback_and_higher_score_wins():
    primary = [R("table", 0.3, (0, 0, 10, 10))]  # below min -> fallback fires
    fb = [R("table", 0.85, (0, 0, 10, 10), source="table_fallback")]  # overlaps, higher score
    out = detect_layout(None, lambda img: primary, lambda img: fb, min_table_score=0.5, dedup_iou=0.5)
    tables = [r for r in out if r.label == "table"]
    assert len(tables) == 1
    assert tables[0].score == 0.85 and tables[0].source == "table_fallback"


# --- detect_layout: fallback contributes tables only ---


def test_fallback_contributes_tables_only():
    # Primary has a low-score table (triggers fallback); fallback's non-table region must be dropped
    primary = [R("text", 0.9, (0, 0, 10, 10)), R("table", 0.2, (1, 1, 9, 9))]
    fb = [
        R("table", 0.8, (1, 1, 9, 9), source="table_fallback"),
        R("picture", 0.95, (0, 0, 10, 10), source="table_fallback"),  # dropped: non-table
    ]
    out = detect_layout(None, lambda img: primary, lambda img: fb)
    assert sorted(r.label for r in out) == ["table", "text"]


# --- detect_layout: dedup confined to table regions ---


def test_dedup_removes_duplicate_table_but_keeps_overlapping_nontable():
    # a picture fully overlaps two duplicate tables; tables dedup to one, the picture survives
    primary = [
        R("picture", 0.99, (0, 0, 10, 10)),
        R("table", 0.6, (0, 0, 10, 10)),  # strong table -> no fallback
        R("table", 0.55, (0, 0, 10, 10)),  # duplicate -> deduped away
    ]
    out = detect_layout(None, lambda img: primary, min_table_score=0.5, dedup_iou=0.5)
    assert sorted(r.label for r in out) == ["picture", "table"]
    assert next(r for r in out if r.label == "table").score == 0.6
    assert any(r.label == "picture" for r in out)  # not suppressed by the overlapping tables


def test_dedup_keeps_disjoint_tables():
    primary = [R("table", 0.9, (0, 0, 10, 10)), R("table", 0.8, (50, 50, 60, 60))]
    out = detect_layout(None, lambda img: primary)
    assert len([r for r in out if r.label == "table"]) == 2


def test_dedup_table_tie_keeps_primary_deterministically():
    # equal-score overlapping primary (low) + fallback tables -> primary (lower index) wins the tie
    primary = [R("table", 0.4, (0, 0, 10, 10), source="layout")]  # below min -> fallback fires
    fb = [R("table", 0.4, (0, 0, 10, 10), source="table_fallback")]
    out = detect_layout(None, lambda img: primary, lambda img: fb, min_table_score=0.5, dedup_iou=0.5)
    tables = [r for r in out if r.label == "table"]
    assert len(tables) == 1
    assert tables[0].source == "layout"
