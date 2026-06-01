"""Phase 1B (Colab CPU/GPU runtime): fill TATR-predicted grids with PaddleOCR text.

OCR currently runs on CPU PaddlePaddle (requirements-colab pins `paddlepaddle`, not
`paddlepaddle-gpu`), so this step does not need a GPU runtime - only Phase 1A TATR
inference did. Triggered from the notebook:
    !python scripts/run_phase1b_ocr_filled.py --limit 10 --seed 42 --run-id debug

For each sample (using the same seeded subset convention as Phase 1A) this:
  1. loads the Phase 1A TATR-predicted canonical grid (outputs/tables/tatr_predicted/),
  2. runs PaddleOCR on the table crop,
  3. assigns OCR words to the predicted cells (the same assign_words_to_cells used for
     gt_filled),
  4. writes the filled table to outputs/tables/ocr_filled/ (text_source="ocr"), records
     a manifest row, and logs failures.

This is the real extraction output: TATR-predicted topology + PaddleOCR text. It is
compared against gt_filled (GT topology + GT text) in the content evaluation; the two are
built by the same fill path so the metrics isolate topology and OCR-text differences.

Requires Phase 1A predictions to exist for the subset (samples without a tatr_predicted
file are skipped, not failed). Resumable: re-running skips sample_ids already success.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.canonical_schema import EVAL_TYPE_CONTENT, TEXT_SOURCE_OCR  # noqa: E402
from src.failure_logger import FailureLogger  # noqa: E402
from src.fintabnet_loader import (  # noqa: E402
    download_structure,
    find_xml_files,
    image_path_for,
    structure_root,
)
from src.ocr_adapter import build_paddleocr, run_paddleocr  # noqa: E402
from src.run_manifest import STATUS_FAILED, STATUS_SUCCESS, RunManifest  # noqa: E402
from src.tatr_postprocess import assign_words_to_cells  # noqa: E402

PHASE = "phase1b_ocr"


def _fail(failures, manifest, sample_id, pred_path, error_type, message) -> None:
    failures.log(sample_id, PHASE, error_type, message)
    manifest.record(
        sample_id, STATUS_FAILED, input_path=str(pred_path), error_type=error_type
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10, help="max samples (0 = all)")
    ap.add_argument("--run-id", default="debug")
    ap.add_argument("--seed", type=int, default=None,
                    help="random-sample seed (nested across limits); omit for first-N")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    from PIL import Image

    download_structure(force=args.force_download)
    root = structure_root()
    xmls = find_xml_files(root, limit=args.limit or None, seed=args.seed)

    manifest = RunManifest(config.MANIFESTS / f"{PHASE}_{args.run_id}.csv")
    failures = FailureLogger(config.FAILURE_LOGS / f"{PHASE}_{args.run_id}.jsonl")
    pred_dir = config.TABLES_TATR_PREDICTED
    out_dir = config.TABLES_OCR_FILLED
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = build_paddleocr()

    processed = skipped = failed = no_prediction = 0
    words_total = words_assigned = 0
    for xml in xmls:
        sample_id = xml.stem
        if manifest.is_done(sample_id):
            skipped += 1
            continue

        pred_path = pred_dir / f"{sample_id}.json"
        if not pred_path.exists():
            # No Phase 1A prediction for this sample yet; not a failure of this phase.
            no_prediction += 1
            continue

        try:
            table = json.loads(pred_path.read_text(encoding="utf-8"))
            image = Image.open(image_path_for(sample_id)).convert("RGB")
        except Exception as e:  # noqa: BLE001 - one bad sample must not abort the batch
            _fail(failures, manifest, sample_id, pred_path, "annotation_format",
                  f"load failed: {e}")
            failed += 1
            continue

        try:
            words = [w.to_dict() for w in run_paddleocr(image, ocr)]
        except Exception as e:  # noqa: BLE001
            _fail(failures, manifest, sample_id, pred_path, "ocr",
                  f"PaddleOCR failed: {e}")
            failed += 1
            continue

        assign_words_to_cells(table["cells"], words, sample_id=sample_id)
        table["meta"] = {
            "sample_id": sample_id,
            "text_source": TEXT_SOURCE_OCR,
            "evaluation_type": EVAL_TYPE_CONTENT,
        }
        out_path = out_dir / f"{sample_id}.json"
        out_path.write_text(json.dumps(table), encoding="utf-8")

        words_total += len(words)
        words_assigned += sum(len(c.get("words", [])) for c in table["cells"])
        manifest.record(
            sample_id, STATUS_SUCCESS,
            input_path=str(pred_path), output_path=str(out_path),
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
        "no_prediction": no_prediction,
        "words_total": words_total,
        "words_assigned": words_assigned,
        "word_assignment_coverage": coverage,
    }
    runlog_path = config.MANIFESTS / f"{PHASE}_runs.jsonl"
    runlog_path.parent.mkdir(parents=True, exist_ok=True)
    with runlog_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary) + "\n")

    print(f"processed={processed} skipped={skipped} failed={failed} "
          f"no_prediction={no_prediction}")
    print(f"OCR word assignment coverage: {coverage} "
          f"({words_assigned}/{words_total})")
    print(f"ocr_filled -> {out_dir}")
    print(f"run log    -> {runlog_path}")


if __name__ == "__main__":
    main()
