"""Phase 1C (CPU): build retrieval chunks from filled tables.

Triggered from the notebook:
    !python scripts/build_table_chunks.py

For each filled table this serializes it (markdown and linearized) and writes one chunk
per table to a JSONL corpus per (text_source, serialization):

    outputs/rag_index/chunks/{gt,ocr}_{markdown,linearized}.jsonl

GT-filled and OCR-filled corpora are kept strictly separate (P4); the retrieval +
QA experiment runs over each independently so the OCR-vs-GT downstream gap is measurable.
Pure CPU, no model. Re-running overwrites the corpus files (deterministic from the tables).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.table_chunk import build_chunk  # noqa: E402
from src.table_serialize import SERIALIZATIONS  # noqa: E402

SOURCE_DIRS = {
    "gt": config.TABLES_GT_FILLED,
    "ocr": config.TABLES_OCR_FILLED,
}


def _build_corpus(source: str, serialization: str) -> tuple[int, int]:
    """Write one chunk per table for (source, serialization). Returns (written, empty)."""
    src_dir = SOURCE_DIRS[source]
    out_path = config.CHUNKS / f"{source}_{serialization}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = empty = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for table_path in sorted(src_dir.glob("*.json")):
            table = json.loads(table_path.read_text(encoding="utf-8"))
            chunk = build_chunk(table, serialization=serialization)
            if not chunk["text"].strip():
                # An empty table serializes to nothing; it cannot be retrieved, so skip it.
                empty += 1
                continue
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            written += 1
    return written, empty


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["gt", "ocr", "both"], default="both")
    ap.add_argument("--serialization", choices=[*SERIALIZATIONS, "both"], default="both")
    args = ap.parse_args()

    sources = ["gt", "ocr"] if args.source == "both" else [args.source]
    serials = list(SERIALIZATIONS) if args.serialization == "both" else [args.serialization]

    for source in sources:
        if not SOURCE_DIRS[source].exists():
            print(f"[skip] {source}: no tables dir at {SOURCE_DIRS[source]}")
            continue
        for serialization in serials:
            written, empty = _build_corpus(source, serialization)
            out_path = config.CHUNKS / f"{source}_{serialization}.jsonl"
            print(f"{source:>3} / {serialization:<10} -> {written} chunks "
                  f"({empty} empty skipped)  {out_path}")


if __name__ == "__main__":
    main()
