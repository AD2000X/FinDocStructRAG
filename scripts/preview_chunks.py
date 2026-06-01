"""Phase 1C (CPU): browse filled tables as readable markdown for manual QA authoring.

Triggered from the notebook:
    !python scripts/preview_chunks.py --limit 15 --seed 7

Prints a seeded sample of GT-filled tables rendered as markdown (which reads like a real
table), with their sample_id, so you can read off true answers and write the manual +
unanswerable questions for qa/qa_manual_seed.jsonl. GT-filled is used because gold answers
come from GT (P4). Run in the notebook to also show each crop alongside its markdown.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.table_chunk import chunk_id_for  # noqa: E402
from src.table_serialize import serialize_markdown  # noqa: E402

SOURCE_DIRS = {"gt": config.TABLES_GT_FILLED, "ocr": config.TABLES_OCR_FILLED}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["gt", "ocr"], default="gt")
    ap.add_argument("--limit", type=int, default=15, help="how many tables to show")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    src_dir = SOURCE_DIRS[args.source]
    if not src_dir.exists():
        raise SystemExit(f"no {args.source}_filled tables at {src_dir}")

    paths = sorted(src_dir.glob("*.json"))
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.limit]

    for path in paths:
        table = json.loads(path.read_text(encoding="utf-8"))
        sample_id = table.get("meta", {}).get("sample_id", path.stem)
        print("=" * 100)
        print(f"sample_id : {sample_id}")
        print(f"chunk_id  : {chunk_id_for(sample_id)}   "
              f"({table.get('num_rows', 0)}x{table.get('num_cols', 0)})")
        print(serialize_markdown(table) or "(empty)")
        print()


if __name__ == "__main__":
    main()
