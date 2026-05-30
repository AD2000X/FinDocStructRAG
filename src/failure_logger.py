"""Failure logging (DESIGN_SPEC §10).

Records per-sample failures during a batch run so a single bad sample does not abort
the whole pipeline and failures can be reviewed afterwards. Writes JSONL (append-only,
resume-friendly) and can dump a CSV summary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Error type taxonomy. Extend as phases add new failure modes; keeping the set named
# here means metrics group failures consistently.
ERROR_TYPES = (
    "annotation_format",   # FinTabNet.c annotation cannot convert to canonical
    "grid_geometry",       # validate_grid_geometry rejected the grid
    "html_parse",          # html_to_canonical failed
    "tatr_inference",      # TATR model call failed
    "ocr",                 # OCR step failed
    "word_assignment",     # assign_words_to_cells could not place words
    "unknown",
)


@dataclass
class FailureRecord:
    sample_id: str
    phase: str
    error_type: str
    message: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class FailureLogger:
    """Append-only failure log for one phase/run."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.records: list[FailureRecord] = []

    def log(
        self,
        sample_id: str,
        phase: str,
        error_type: str = "unknown",
        message: str = "",
    ) -> FailureRecord:
        if error_type not in ERROR_TYPES:
            error_type = "unknown"
        record = FailureRecord(sample_id, phase, error_type, message)
        self.records.append(record)
        self._append(record)
        return record

    def _append(self, record: FailureRecord) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def __len__(self) -> int:
        return len(self.records)
