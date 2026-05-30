"""Phase 1A (Colab GPU): TATR structure inference + topology metrics.

Triggered from notebooks/01_phase1a_tatr.ipynb:
    !python scripts/run_phase1a_colab.py --limit 50 --run-id debug

For each FinTabNet.c sample this:
  1. parses the GT PASCAL VOC structure annotation (fintabnet_loader),
  2. runs TATR structure recognition on the table crop (GPU),
  3. derives a canonical grid from the predicted row/col/spanning boxes,
  4. compares predicted topology against GT topology (eval_table),
  5. writes the prediction to outputs/tables/tatr_predicted/, records a manifest row,
     and logs failures.

P4: only the TATR prediction is persisted (text_source=none). GT text is not written;
gt_filled/ is reserved for GT-text-filled tables in Phase 1B. The GT used here is the
annotation's topology, re-derived from the XML each run.

Resumable: re-running skips sample_ids already marked success in the manifest. The
topology report covers the samples processed in the current run; recomputing metrics
over all persisted predictions belongs in scripts/evaluate_tables.py.

GPU inference lives here in the runner (P1: scripts/ may hold logic; notebooks only
call it). The pure post-processing and metrics it calls are unit-tested in tests/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.canonical_schema import EVAL_TYPE_TOPOLOGY, TEXT_SOURCE_NONE  # noqa: E402
from src.eval_table import (  # noqa: E402
    aggregate_topology,
    topology_sample_metrics,
    write_topology_report,
)
from src.failure_logger import FailureLogger  # noqa: E402
from src.fintabnet_loader import (  # noqa: E402
    download_structure,
    find_xml_files,
    parse_structure_xml,
    structure_root,
)
from src.run_manifest import STATUS_FAILED, STATUS_SUCCESS, RunManifest  # noqa: E402
from src.tatr_postprocess import (  # noqa: E402
    normalize_tatr_prediction,
    validate_grid_geometry,
)

PHASE = "phase1a"

# Predicted structure-model label -> our prediction-dict key. Other classes
# (table, column/row header) are ignored for topology grid derivation.
LABEL_TO_KEY = {
    "table row": "row_boxes",
    "table column": "col_boxes",
    "table spanning cell": "spanning_cells",
}


def _load_model(device):
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    # use_fast=False: the fast DETR processor raises "'SizeDict' object has no
    # attribute 'keys'" during post-processing; the slow processor is stable for TATR.
    processor = AutoImageProcessor.from_pretrained(
        config.TATR_STRUCTURE_MODEL, use_fast=False
    )
    # This checkpoint ships size={'longest_edge': N} only, which the resize step
    # rejects (it needs shortest_edge+longest_edge, or height+width). Add a
    # shortest_edge while preserving the checkpoint's longest_edge.
    longest = processor.size.get("longest_edge", 1000)
    processor.size = {"shortest_edge": min(800, longest), "longest_edge": longest}
    model = TableTransformerForObjectDetection.from_pretrained(
        config.TATR_STRUCTURE_MODEL
    )
    model.to(device).eval()
    return processor, model


def _infer_boxes(processor, model, device, image, threshold) -> dict:
    """Run TATR and group predicted boxes by class into a prediction dict."""
    import torch

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    result = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    id2label = model.config.id2label

    pred = {"row_boxes": [], "col_boxes": [], "spanning_cells": []}
    for label_id, box in zip(result["labels"].tolist(), result["boxes"].tolist()):
        key = LABEL_TO_KEY.get(id2label[label_id])
        if key:
            pred[key].append({"bbox": [float(v) for v in box]})
    return pred


def _image_index(root) -> dict:
    """Map image filename -> path, so a prediction can find its crop by name."""
    return {p.name: p for p in root.rglob("*.jpg")}


def _fail(failures, manifest, sample_id, xml, error_type, message) -> None:
    failures.log(sample_id, PHASE, error_type, message)
    manifest.record(
        sample_id, STATUS_FAILED, input_path=str(xml), error_type=error_type
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50, help="max samples (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--run-id", default="debug")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    download_structure(force=args.force_download)
    root = structure_root()
    xmls = find_xml_files(root, limit=args.limit or None)
    images = _image_index(root)

    manifest = RunManifest(config.MANIFESTS / f"{PHASE}_{args.run_id}.csv")
    failures = FailureLogger(config.FAILURE_LOGS / f"{PHASE}_{args.run_id}.jsonl")
    pred_dir = config.TABLES_TATR_PREDICTED
    pred_dir.mkdir(parents=True, exist_ok=True)

    processor, model = _load_model(device)

    per_sample: list[dict] = []
    processed = skipped = failed = 0
    for xml in xmls:
        sample_id = xml.stem
        if manifest.is_done(sample_id):
            skipped += 1
            continue

        try:
            gt = parse_structure_xml(xml)
        except Exception as e:  # noqa: BLE001 - one bad sample must not abort the batch
            _fail(failures, manifest, sample_id, xml, "annotation_format",
                  f"XML parse failed: {e}")
            failed += 1
            continue
        gt_table = normalize_tatr_prediction(gt)

        image_name = gt["image_filename"] or f"{sample_id}.jpg"
        image_path = images.get(image_name)
        if image_path is None:
            _fail(failures, manifest, sample_id, xml, "annotation_format",
                  f"image not found: {image_name}")
            failed += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            pred = _infer_boxes(processor, model, device, image, args.threshold)
        except Exception as e:  # noqa: BLE001
            _fail(failures, manifest, sample_id, xml, "tatr_inference",
                  f"TATR inference failed: {e}")
            failed += 1
            continue

        pred_table = normalize_tatr_prediction(pred)
        # Validate the same sorted boxes the grid is built from; raw TATR order would
        # trip the sort check. Geometry issues are quality flags, not sample failures.
        rows_sorted = sorted(pred["row_boxes"], key=lambda r: r["bbox"][1])
        cols_sorted = sorted(pred["col_boxes"], key=lambda c: c["bbox"][0])
        validate_grid_geometry(
            rows_sorted, cols_sorted, pred_table["cells"],
            logger=failures, sample_id=sample_id,
        )
        pred_table["meta"] = {
            "sample_id": sample_id,
            "text_source": TEXT_SOURCE_NONE,
            "evaluation_type": EVAL_TYPE_TOPOLOGY,
        }
        out_path = pred_dir / f"{sample_id}.json"
        out_path.write_text(json.dumps(pred_table), encoding="utf-8")

        per_sample.append(topology_sample_metrics(pred_table, gt_table))
        manifest.record(
            sample_id, STATUS_SUCCESS,
            input_path=str(xml), output_path=str(out_path),
        )
        processed += 1

    summary = aggregate_topology(per_sample)
    report_path = write_topology_report(
        config.EVALUATION / f"{PHASE}_topology_{args.run_id}.json", summary
    )

    # Append a one-line run summary (the run journal). The presence of a line means the
    # run completed; per-sample failures are detailed in the failure log.
    run_summary = {
        "run_id": args.run_id,
        "phase": PHASE,
        "status": "completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        "threshold": args.threshold,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "metrics": summary,
    }
    runlog_path = config.MANIFESTS / f"{PHASE}_runs.jsonl"
    runlog_path.parent.mkdir(parents=True, exist_ok=True)
    with runlog_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary) + "\n")

    print(f"processed={processed} skipped={skipped} failed={failed}")
    print(f"topology report -> {report_path}")
    print(f"run log     -> {runlog_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
