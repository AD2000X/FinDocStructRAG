"""Phase 1C (CPU): backfill is_header on ocr_filled tables from the TATR raw artifact.

ocr_filled tables were built from tatr_predicted, whose canonical grid carries no header
marking, so every ocr_filled cell has is_header=False. gt_filled, in contrast, was
regenerated with column-header marking. That asymmetry makes the linearized serialization
unfair to compare across sources: gt_linearized pairs each value with its column header
while ocr_linearized cannot, so the two are different serializations, not GT-vs-OCR text.

This is a fairness fix, not an optimization, and it does NOT re-run OCR. For each ocr_filled
table it reads the column_headers boxes from the matching tatr_raw/<sample_id>.json (same
TATR coordinate space as the predicted grid) and applies the same IoMin marking used for
gt_filled (_mark_column_headers). Idempotent: re-running re-marks the same cells.

    !python scripts/mark_ocr_filled_headers.py

Re-build the OCR corpora (build_table_chunks.py) and re-score (evaluate_rag.py) afterwards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.tatr_postprocess import _mark_column_headers  # noqa: E402


def mark_table_headers(table: dict, column_headers: list[dict]) -> int:
    """Mark is_header on the table's cells from column-header boxes; return count marked."""
    cells = table.get("cells", [])
    _mark_column_headers(cells, column_headers)
    return sum(1 for c in cells if c.get("is_header"))


def main() -> None:
    ocr_dir = config.TABLES_OCR_FILLED
    raw_dir = config.TABLES_TATR_RAW
    ocr_files = sorted(ocr_dir.glob("*.json"))
    if not ocr_files:
        raise SystemExit(f"no ocr_filled tables at {ocr_dir}")

    patched = no_raw = no_headers = 0
    for path in ocr_files:
        sample_id = path.stem
        raw_path = raw_dir / f"{sample_id}.json"
        if not raw_path.exists():
            no_raw += 1
            continue
        column_headers = json.loads(raw_path.read_text(encoding="utf-8")).get(
            "column_headers", []
        )
        if not column_headers:
            no_headers += 1
            continue

        table = json.loads(path.read_text(encoding="utf-8"))
        marked = mark_table_headers(table, column_headers)
        path.write_text(json.dumps(table), encoding="utf-8")
        if marked:
            patched += 1
        else:
            no_headers += 1

    print(f"ocr_filled tables : {len(ocr_files)}")
    print(f"patched (>=1 hdr) : {patched}")
    print(f"no tatr_raw       : {no_raw}")
    print(f"no header cells   : {no_headers}")
    print(f"ocr_filled -> {ocr_dir}")


if __name__ == "__main__":
    main()
