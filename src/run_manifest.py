"""Resumable run manifest (DESIGN_SPEC §18.2).

One row per sample, keyed by sample_id, appended and flushed per sample so a
mid-run crash (e.g. a dropped Colab session) still leaves a resumable record. On
restart, construct a RunManifest over the same path and skip every sample_id that
is already marked success (is_done).

This is a thin, pure-CPU artifact primitive; metric computation lives in
eval_table.py and failure detail lives in failure_logger.py.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUSES = (STATUS_SUCCESS, STATUS_FAILED, STATUS_SKIPPED)

# Fixed column order (DESIGN_SPEC §18.2). Do not extend without a spec bump.
FIELDNAMES = (
    "sample_id",
    "status",
    "input_path",
    "output_path",
    "error_type",
    "timestamp",
)


@dataclass
class ManifestRecord:
    sample_id: str
    status: str
    input_path: str = ""
    output_path: str = ""
    error_type: str = ""   # "" on success; failure taxonomy (§10) on failure
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RunManifest:
    """Append-only, resumable manifest for one batch run."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self._succeeded: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Rebuild the succeeded set from an existing manifest (resume)."""
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == STATUS_SUCCESS:
                    self._succeeded.add(row["sample_id"])

    def is_done(self, sample_id: str) -> bool:
        """True if sample_id already completed successfully (skip on resume)."""
        return sample_id in self._succeeded

    @property
    def completed_ids(self) -> set[str]:
        return set(self._succeeded)

    def record(
        self,
        sample_id: str,
        status: str,
        input_path: str = "",
        output_path: str = "",
        error_type: str = "",
    ) -> ManifestRecord:
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status!r}")
        rec = ManifestRecord(
            sample_id, status, input_path, output_path, error_type
        )
        self._append(rec)
        if status == STATUS_SUCCESS:
            self._succeeded.add(sample_id)
        return rec

    def _append(self, record: ManifestRecord) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.manifest_path.exists()
        with self.manifest_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(record))
            f.flush()
