"""Unit tests for src/bbox_utils (pure geometry; crop tests skip if Pillow is absent)."""

import pytest

from src import bbox_utils as bu


# --- format conversions ---


def test_xyxy_xywh_roundtrip():
    box = (10.0, 20.0, 50.0, 80.0)
    assert bu.xyxy_to_xywh(box) == (10.0, 20.0, 40.0, 60.0)
    assert bu.xywh_to_xyxy(bu.xyxy_to_xywh(box)) == box


def test_xywh_to_xyxy():
    assert bu.xywh_to_xyxy((10.0, 20.0, 40.0, 60.0)) == (10.0, 20.0, 50.0, 80.0)


# --- clamp ---


def test_clamp_box_inside_unchanged():
    box = (10.0, 10.0, 90.0, 90.0)
    assert bu.clamp_box(box, 100, 100) == box


def test_clamp_box_negative_and_overflow():
    # detector boxes can run negative / past the page edge (seen in the smoke)
    assert bu.clamp_box((-20.0, -5.0, 120.0, 130.0), 100, 100) == (0.0, 0.0, 100.0, 100.0)


def test_clamp_box_fully_outside_collapses():
    # entirely left of the page -> zero-width after clamp; caller detects via box_area
    clamped = bu.clamp_box((-50.0, 10.0, -10.0, 40.0), 100, 100)
    assert clamped == (0.0, 10.0, 0.0, 40.0)
    assert bu.box_area(clamped) == 0.0


# --- area ---


def test_box_area_normal_and_degenerate():
    assert bu.box_area((0.0, 0.0, 10.0, 5.0)) == 50.0
    assert bu.box_area((10.0, 0.0, 0.0, 5.0)) == 0.0  # inverted x -> 0, never negative


# --- iou ---


def test_iou_identical():
    box = (0.0, 0.0, 10.0, 10.0)
    assert bu.iou(box, box) == 1.0


def test_iou_disjoint():
    assert bu.iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_partial_overlap():
    # two 10x10 boxes sharing a 5x5 corner: inter=25, union=200-25=175
    assert bu.iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)) == pytest.approx(25.0 / 175.0)


def test_iou_containment():
    outer = (0.0, 0.0, 10.0, 10.0)
    inner = (2.0, 2.0, 4.0, 4.0)  # area 4 fully inside area 100
    assert bu.iou(outer, inner) == pytest.approx(4.0 / 100.0)


# --- dedup ---


def test_dedup_keeps_higher_score():
    boxes = [(0.0, 0.0, 10.0, 10.0), (1.0, 1.0, 11.0, 11.0)]  # heavy overlap
    assert bu.dedup_boxes(boxes, [0.9, 0.5], iou_threshold=0.5) == [0]


def test_dedup_keeps_disjoint_in_score_order():
    boxes = [(0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0)]
    assert bu.dedup_boxes(boxes, [0.5, 0.9], iou_threshold=0.5) == [1, 0]


def test_dedup_tie_break_by_index():
    # identical boxes, equal scores -> lower original index wins, deterministic
    boxes = [(0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)]
    assert bu.dedup_boxes(boxes, [0.7, 0.7], iou_threshold=0.5) == [0]


def test_dedup_below_threshold_keeps_both():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)  # IoU ~= 0.143 < 0.5
    assert set(bu.dedup_boxes([a, b], [0.9, 0.8], iou_threshold=0.5)) == {0, 1}


def test_dedup_length_mismatch_raises():
    with pytest.raises(ValueError):
        bu.dedup_boxes([(0.0, 0.0, 1.0, 1.0)], [0.5, 0.6], 0.5)


# --- crop_with_padding (needs Pillow; skipped locally if absent, runs on Colab) ---


def _img(w, h):
    image_mod = pytest.importorskip("PIL.Image")
    return image_mod.new("RGB", (w, h), "white")


def test_crop_basic_size():
    crop = bu.crop_with_padding(_img(100, 100), (10.0, 20.0, 40.0, 60.0))
    assert crop.size == (30, 40)


def test_crop_padding_expands():
    # floor(10-5,20-5)=(5,15); ceil(40+5,60+5)=(45,65) -> 40 x 50
    crop = bu.crop_with_padding(_img(100, 100), (10.0, 20.0, 40.0, 60.0), pad=5.0)
    assert crop.size == (40, 50)


def test_crop_floor_ceil_rounding():
    # floor(10.4,20.6)=(10,20); ceil(40.1,60.9)=(41,61) -> 31 x 41
    crop = bu.crop_with_padding(_img(100, 100), (10.4, 20.6, 40.1, 60.9))
    assert crop.size == (31, 41)


def test_crop_clamps_to_image():
    crop = bu.crop_with_padding(_img(50, 50), (-10.0, -10.0, 80.0, 80.0))
    assert crop.size == (50, 50)


def test_crop_degenerate_after_clamp_raises():
    with pytest.raises(ValueError):
        bu.crop_with_padding(_img(50, 50), (60.0, 10.0, 90.0, 40.0))  # entirely right of the image
