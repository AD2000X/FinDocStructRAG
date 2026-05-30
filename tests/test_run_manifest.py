"""Run manifest tests (DESIGN_SPEC §18.2).

Append-and-flush per sample, resume by skipping already-succeeded samples.
"""

import csv

import pytest

from src.run_manifest import (
    FIELDNAMES,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    RunManifest,
    read_completed,
)


def test_columns_match_schema(tmp_path):
    path = tmp_path / "m.csv"
    m = RunManifest(path)
    m.record("s1", STATUS_SUCCESS, input_path="in", output_path="out")
    with path.open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert tuple(header) == FIELDNAMES


def test_record_and_resume(tmp_path):
    path = tmp_path / "m.csv"
    m = RunManifest(path)
    m.record("s1", STATUS_SUCCESS)
    m.record("s2", STATUS_FAILED, error_type="grid_geometry")

    # Re-open over the same path (simulates a restarted Colab session).
    resumed = RunManifest(path)
    assert resumed.is_done("s1") is True
    assert resumed.is_done("s2") is False
    assert resumed.completed_ids == {"s1"}


def test_skipped_is_not_done(tmp_path):
    m = RunManifest(tmp_path / "m.csv")
    m.record("s1", STATUS_SKIPPED)
    assert m.is_done("s1") is False


def test_header_written_once(tmp_path):
    path = tmp_path / "m.csv"
    m = RunManifest(path)
    m.record("s1", STATUS_SUCCESS)
    m.record("s2", STATUS_SUCCESS)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    # 1 header + 2 data rows.
    assert len(rows) == 3
    assert rows.count(list(FIELDNAMES)) == 1


def test_unknown_status_raises(tmp_path):
    m = RunManifest(tmp_path / "m.csv")
    with pytest.raises(ValueError):
        m.record("s1", "bogus")


def test_read_completed_latest_status_wins(tmp_path):
    path = tmp_path / "m.csv"
    m = RunManifest(path)
    m.record("s1", STATUS_FAILED, error_type="tatr_inference")  # earlier failure
    m.record("s1", STATUS_SUCCESS, output_path="out1")          # later success wins
    m.record("s2", STATUS_SUCCESS, output_path="out2")
    m.record("s3", STATUS_FAILED, error_type="grid_geometry")

    rows = read_completed(path)
    assert {r["sample_id"] for r in rows} == {"s1", "s2"}
    s1 = next(r for r in rows if r["sample_id"] == "s1")
    assert s1["output_path"] == "out1"


def test_read_completed_missing_file(tmp_path):
    assert read_completed(tmp_path / "nope.csv") == []
