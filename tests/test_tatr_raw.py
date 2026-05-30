"""Raw TATR artifact tests (CPU, synthetic).

Locks the geometry-flag tee and the raw-artifact schema. The GPU inference that fills
the prediction dict is exercised only on Colab; here we feed synthetic boxes.
"""

from src.tatr_raw import (
    RAW_BOX_KEYS,
    RAW_SCHEMA_VERSION,
    GeometryFlagCollector,
    build_tatr_raw_artifact,
)


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def log(self, sample_id, phase, error_type="unknown", message=""):
        self.calls.append((sample_id, phase, error_type, message))


def test_geometry_collector_records_flags():
    c = GeometryFlagCollector()
    c.log("s1", "phase1a", "grid_geometry", "adjacent rows overlap > 0.3")
    c.log("s1", "phase1a", "grid_geometry", "tiny cell area < 100")
    assert c.flags == ["adjacent rows overlap > 0.3", "tiny cell area < 100"]


def test_geometry_collector_tees_to_delegate():
    delegate = _RecordingLogger()
    c = GeometryFlagCollector(delegate=delegate)
    c.log("s1", "phase1a", "grid_geometry", "flag")
    assert c.flags == ["flag"]
    assert delegate.calls == [("s1", "phase1a", "grid_geometry", "flag")]


def test_build_tatr_raw_artifact_schema():
    pred = {
        "table_boxes": [{"bbox": [0, 0, 10, 10], "score": 0.9, "label": "table"}],
        "row_boxes": [{"bbox": [0, 0, 10, 5], "score": 0.8, "label": "table row"}],
        # col_boxes intentionally omitted -> must default to []
    }
    art = build_tatr_raw_artifact(
        sample_id="s1", image_filename="s1.jpg", prediction=pred,
        geometry_valid=True, geometry_flags=[],
        model_id="m", threshold=0.5, run_id="r",
    )
    assert art["schema_version"] == RAW_SCHEMA_VERSION
    assert art["sample_id"] == "s1"
    assert art["image_filename"] == "s1.jpg"
    assert art["model_id"] == "m"
    assert art["threshold"] == 0.5
    assert art["run_id"] == "r"
    for key in RAW_BOX_KEYS:
        assert key in art
    assert art["col_boxes"] == []
    assert art["geometry_validation"] == {"valid": True, "flags": []}


def test_build_tatr_raw_artifact_keeps_scores_labels_and_flags():
    pred = {"row_boxes": [{"bbox": [0, 0, 10, 5], "score": 0.77, "label": "table row"}]}
    art = build_tatr_raw_artifact(
        sample_id="s", image_filename="s.jpg", prediction=pred,
        geometry_valid=False, geometry_flags=["adjacent rows overlap > 0.3"],
        model_id="m", threshold=0.4, run_id="r",
    )
    box = art["row_boxes"][0]
    assert box["score"] == 0.77
    assert box["label"] == "table row"
    assert art["geometry_validation"] == {
        "valid": False, "flags": ["adjacent rows overlap > 0.3"]
    }
