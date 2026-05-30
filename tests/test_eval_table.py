"""Table topology metric tests (DESIGN_SPEC §6.2).

Synthetic canonical tables. Also exercises the real boxes_to_grid + html_to_canonical
paths once, to confirm the metrics line up with the producers.
"""

from src.eval_table import (
    aggregate_topology,
    occupancy_set,
    spanning_cells_of,
    topology_sample_metrics,
    write_topology_report,
)
from src.tatr_postprocess import boxes_to_grid, html_to_canonical


def _unit_cell(r, c):
    return {"row_start": r, "row_end": r + 1, "col_start": c, "col_end": c + 1,
            "text": "", "is_header": False}


def _grid_2x2():
    return {"num_rows": 2, "num_cols": 2,
            "cells": [_unit_cell(r, c) for r in range(2) for c in range(2)]}


# --- occupancy / spanning helpers -------------------------------------------------

def test_occupancy_set_full_grid():
    assert occupancy_set(_grid_2x2()) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_spanning_cells_of_detects_span():
    table = {"num_rows": 2, "num_cols": 2, "cells": [
        {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 2,
         "text": "H", "is_header": True},
        _unit_cell(1, 0), _unit_cell(1, 1),
    ]}
    assert spanning_cells_of(table) == {(0, 1, 0, 2)}


# --- per-sample metrics -----------------------------------------------------------

def test_perfect_match():
    m = topology_sample_metrics(_grid_2x2(), _grid_2x2())
    assert m["row_count_correct"] is True
    assert m["col_count_correct"] is True
    assert m["cell_occupancy_f1"] == 1.0
    assert m["gt_spanning_cells"] == 0


def test_dimension_mismatch():
    pred = {"num_rows": 3, "num_cols": 2, "cells": _grid_2x2()["cells"]}
    m = topology_sample_metrics(pred, _grid_2x2())
    assert m["row_count_correct"] is False
    assert m["col_count_correct"] is True


def test_spanning_cell_matched_and_missed():
    gt = {"num_rows": 2, "num_cols": 2, "cells": [
        {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 2,
         "text": "H", "is_header": True},
        _unit_cell(1, 0), _unit_cell(1, 1),
    ]}
    # Prediction reproduces the span.
    matched = topology_sample_metrics(gt, gt)
    assert matched["gt_spanning_cells"] == 1
    assert matched["matched_spanning_cells"] == 1
    # Prediction splits the span into unit cells -> missed.
    missed = topology_sample_metrics(_grid_2x2(), gt)
    assert missed["gt_spanning_cells"] == 1
    assert missed["matched_spanning_cells"] == 0


def test_metrics_align_with_producers():
    # GT from the occupancy-aware HTML parser, prediction from boxes_to_grid.
    html = "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"
    gt = html_to_canonical(html)
    rows = [{"bbox": [0, 0, 20, 10]}, {"bbox": [0, 10, 20, 20]}]
    cols = [{"bbox": [0, 0, 10, 20]}, {"bbox": [10, 0, 20, 20]}]
    pred = {"num_rows": 2, "num_cols": 2, "cells": boxes_to_grid(rows, cols)}
    m = topology_sample_metrics(pred, gt)
    assert m["cell_occupancy_f1"] == 1.0
    assert m["row_count_correct"] and m["col_count_correct"]


# --- aggregation ------------------------------------------------------------------

def test_aggregate_empty():
    summary = aggregate_topology([])
    assert summary["num_samples"] == 0
    assert summary["spanning_cell_detection_rate"] is None


def test_aggregate_mixed():
    per_sample = [
        topology_sample_metrics(_grid_2x2(), _grid_2x2()),
        topology_sample_metrics(
            {"num_rows": 3, "num_cols": 2, "cells": _grid_2x2()["cells"]},
            _grid_2x2(),
        ),
    ]
    summary = aggregate_topology(per_sample)
    assert summary["num_samples"] == 2
    assert summary["row_count_accuracy"] == 0.5
    assert summary["col_count_accuracy"] == 1.0
    assert summary["spanning_cell_detection_rate"] is None  # no GT spans in batch


def test_write_report_roundtrip(tmp_path):
    import json
    summary = aggregate_topology([topology_sample_metrics(_grid_2x2(), _grid_2x2())])
    out = write_topology_report(tmp_path / "phase1a_topology.json", summary)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["evaluation_type"] == "topology"
    assert loaded["num_samples"] == 1
