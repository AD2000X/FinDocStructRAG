"""Phase 1B (CPU): build gt_filled tables from GT structure + GT word tokens.

Triggered from the notebook:
    !python scripts/run_phase1b_gt_filled.py --limit 50 --run-id debug --seed 42

For each FinTabNet.c sample this:
  1. parses the GT PASCAL VOC structure annotation (the GT cell grid),
  2. loads the matching GT word tokens (words/<stem>_words.json),
  3. assigns words to grid cells (the same assign_words_to_cells used for OCR),
  4. writes the filled table to outputs/tables/gt_filled/, records a manifest row,
     and logs failures.

gt_filled cell text is reconstructed by assigning FinTabNet.c word-level GT tokens to
the GT structure grid; it is NOT the official cell-level HTML string from
PDF_Annotations. It is used for QA-pipeline validation only and is never reported as an
extraction output (P4, text_source="gt").

No GPU and no crop image needed (text comes from the word JSON, not OCR). Resumable:
re-running skips sample_ids already marked success. ocr_filled (the real extraction
output) reuses fill_table with a TATR-predicted grid and PaddleOCR words.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.canonical_schema import TEXT_SOURCE_GT  # noqa: E402
from src.failure_logger import FailureLogger  # noqa: E402
from src.fintabnet_loader import (  # noqa: E402
    download_structure,
    find_xml_files,
    parse_structure_xml,
    parse_words_json,
    structure_root,
    words_path_for,
)
from src.run_manifest import STATUS_FAILED, STATUS_SUCCESS, RunManifest  # noqa: E402
from src.table_fill import fill_table  # noqa: E402

PHASE = "phase1b_gt"


def _fail(failures, manifest, sample_id, xml, error_type, message) -> None:
    failures.log(sample_id, PHASE, error_type, message)
    manifest.record(
        sample_id, STATUS_FAILED, input_path=str(xml), error_type=error_type
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50, help="max samples (0 = all)")
    ap.add_argument("--run-id", default="debug")
    ap.add_argument("--seed", type=int, default=None,
                    help="random-sample seed (nested across limits); omit for first-N")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    download_structure(force=args.force_download)
    root = structure_root()
    xmls = find_xml_files(root, limit=args.limit or None, seed=args.seed)

    manifest = RunManifest(config.MANIFESTS / f"{PHASE}_{args.run_id}.csv")
    failures = FailureLogger(config.FAILURE_LOGS / f"{PHASE}_{args.run_id}.jsonl")
    out_dir = config.TABLES_GT_FILLED
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = skipped = failed = 0
    words_total = words_assigned = 0
    for xml in xmls:
        sample_id = xml.stem
        if manifest.is_done(sample_id):
            skipped += 1
            continue

        try:
            parsed = parse_structure_xml(xml)
        except Exception as e:  # noqa: BLE001 - one bad sample must not abort the batch
            _fail(failures, manifest, sample_id, xml, "annotation_format",
                  f"XML parse failed: {e}")
            failed += 1
            continue

        words_path = words_path_for(sample_id)
        if not words_path.exists():
            _fail(failures, manifest, sample_id, xml, "annotation_format",
                  f"words json not found: {words_path.name}")
            failed += 1
            continue

        try:
            words = parse_words_json(words_path)
        except Exception as e:  # noqa: BLE001
            _fail(failures, manifest, sample_id, xml, "annotation_format",
                  f"words json parse failed: {e}")
            failed += 1
            continue

        table = fill_table(
            parsed, words, sample_id=sample_id, text_source=TEXT_SOURCE_GT)
        out_path = out_dir / f"{sample_id}.json"
        out_path.write_text(json.dumps(table), encoding="utf-8")

        words_total += len(words)
        words_assigned += sum(len(c.get("words", [])) for c in table["cells"])
        manifest.record(
            sample_id, STATUS_SUCCESS,
            input_path=str(xml), output_path=str(out_path),
        )
        processed += 1

    coverage = words_assigned / words_total if words_total else None
    run_summary = {
        "run_id": args.run_id,
        "phase": PHASE,
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        "seed": args.seed,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "words_total": words_total,
        "words_assigned": words_assigned,
        "word_assignment_coverage": coverage,
    }
    runlog_path = config.MANIFESTS / f"{PHASE}_runs.jsonl"
    runlog_path.parent.mkdir(parents=True, exist_ok=True)
    with runlog_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary) + "\n")

    print(f"processed={processed} skipped={skipped} failed={failed}")
    print(f"GT word assignment coverage: {coverage} "
          f"({words_assigned}/{words_total})")
    print(f"gt_filled -> {out_dir}")
    print(f"run log   -> {runlog_path}")


if __name__ == "__main__":
    main()
