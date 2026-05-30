"""Raw TATR prediction artifact (Phase 1A debug stream).

The Colab runner derives a canonical topology grid from TATR's output and writes it to
tatr_predicted/ (the thing topology metrics score). That canonical grid drops the raw
predicted boxes, scores, labels, and the header classes - which the deliverable
visualisations (TATR box overlay, grid-geometry report) need. This module assembles a
separate raw artifact so those can be drawn later without re-running the GPU model, and
without mixing prediction with GT (P4).

Pure CPU, no torch: unit-tested in tests/. The runner is the only caller.
"""

from __future__ import annotations

RAW_SCHEMA_VERSION = 1

# TATR structure id2label -> raw-artifact key. Unlike the topology path, this keeps the
# header classes too (they carry no is_header signal into the grid yet, but the overlay
# needs them).
RAW_LABEL_TO_KEY = {
    "table": "table_boxes",
    "table row": "row_boxes",
    "table column": "col_boxes",
    "table column header": "column_headers",
    "table projected row header": "projected_row_headers",
    "table spanning cell": "spanning_cells",
}

RAW_BOX_KEYS = tuple(RAW_LABEL_TO_KEY.values())


class GeometryFlagCollector:
    """Drop-in for validate_grid_geometry's logger that tees its flags.

    Matches FailureLogger.log's signature, so it can be passed straight in. Each flag
    message is collected (for the raw artifact's geometry_validation) and, if a delegate
    FailureLogger is given, still forwarded to it (so the failure log is unchanged).
    """

    def __init__(self, delegate=None):
        self.delegate = delegate
        self.flags: list[str] = []

    def log(self, sample_id, phase, error_type="unknown", message=""):
        self.flags.append(message)
        if self.delegate is not None:
            self.delegate.log(sample_id, phase, error_type, message)


def build_tatr_raw_artifact(
    *,
    sample_id: str,
    image_filename: str,
    prediction: dict,
    geometry_valid: bool,
    geometry_flags: list[str],
    model_id: str,
    threshold: float,
    run_id: str,
) -> dict:
    """Assemble the raw artifact dict (schema RAW_SCHEMA_VERSION).

    prediction is the grouped-boxes dict from the runner's inference step; each box is
    {"bbox": [...], "score": float, "label": str}. Missing classes default to [].
    """
    artifact = {
        "schema_version": RAW_SCHEMA_VERSION,
        "sample_id": sample_id,
        "image_filename": image_filename,
        "model_id": model_id,
        "threshold": threshold,
        "run_id": run_id,
    }
    for key in RAW_BOX_KEYS:
        artifact[key] = prediction.get(key, [])
    artifact["geometry_validation"] = {
        "valid": geometry_valid,
        "flags": geometry_flags,
    }
    return artifact
