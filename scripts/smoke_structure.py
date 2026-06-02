#!/usr/bin/env python3
"""Smoke: Phase 2 crop → TATR structure recognition handoff check.

Picks up to --n crops from LAYOUT_OUTPUT/crops/ (Phase 2 output), runs each
through the full structure pipeline (model inference + normalize + validate),
and prints a one-line summary per crop. No artifacts written; purpose is only
to confirm the crop format is compatible with the Phase 1 structure model.

GPU required (T4 on Colab). CPU fallback works but is slow.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

from src import config
from src.tatr_postprocess import normalize_tatr_prediction, validate_grid_geometry
from src.tatr_raw import RAW_BOX_KEYS, RAW_LABEL_TO_KEY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke: Phase 2 crops → TATR structure")
    p.add_argument("--crops-dir", type=Path, default=None,
                   help="directory of crop PNGs (default: config.LAYOUT_OUTPUT/crops)")
    p.add_argument("--n", type=int, default=5, help="number of crops to test")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="TATR detection threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    crops_dir = args.crops_dir or config.LAYOUT_OUTPUT / "crops"
    crops = sorted(crops_dir.glob("*.png"))[: args.n]
    if not crops:
        print(f"[smoke] no crops found in {crops_dir}")
        return
    print(f"[smoke] {len(crops)} crops from {crops_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] loading {config.TATR_STRUCTURE_MODEL} on {device} ...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(
        config.TATR_STRUCTURE_MODEL, use_fast=False
    )
    # Size dict fix: checkpoint only has longest_edge; add shortest_edge so resize works.
    longest = processor.size.get("longest_edge", 1000)
    processor.size = {"shortest_edge": min(800, longest), "longest_edge": longest}
    model = TableTransformerForObjectDetection.from_pretrained(
        config.TATR_STRUCTURE_MODEL
    ).to(device).eval()
    print(f"[smoke] model ready in {time.time() - t0:.1f}s")

    passed = warned = 0
    for crop_path in crops:
        img = Image.open(crop_path).convert("RGB")

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        result = processor.post_process_object_detection(
            outputs, threshold=args.threshold, target_sizes=target_sizes
        )[0]

        id2label = model.config.id2label
        pred: dict = {k: [] for k in RAW_BOX_KEYS}
        for label_id, score, box in zip(
            result["labels"].tolist(),
            result["scores"].tolist(),
            result["boxes"].tolist(),
        ):
            key = RAW_LABEL_TO_KEY.get(id2label[label_id])
            if key:
                pred[key].append({"bbox": [float(v) for v in box],
                                   "score": float(score),
                                   "label": id2label[label_id]})

        canonical = normalize_tatr_prediction(pred)
        rows_sorted = sorted(pred["row_boxes"], key=lambda r: r["bbox"][1])
        cols_sorted = sorted(pred["col_boxes"], key=lambda c: c["bbox"][0])
        valid = validate_grid_geometry(
            rows_sorted, cols_sorted, canonical["cells"], sample_id=crop_path.stem
        )

        status = "OK  " if valid else "WARN"
        if valid:
            passed += 1
        else:
            warned += 1
        print(
            f"  {status}  {crop_path.name:<45}"
            f"  rows={canonical['num_rows']:>2}  cols={canonical['num_cols']:>2}"
            f"  cells={len(canonical['cells']):>3}  valid={valid}"
        )

    print(f"\n[smoke] {passed} OK / {warned} WARN out of {len(crops)}")
    if warned == 0:
        print("[smoke] structure handoff OK")


if __name__ == "__main__":
    main()
