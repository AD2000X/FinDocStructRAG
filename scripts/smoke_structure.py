#!/usr/bin/env python3
"""Smoke: Phase 2 crop → TATR structure recognition handoff check.

Picks up to --n crops from LAYOUT_OUTPUT/crops/ (Phase 2 output), runs each
through the full structure pipeline (model inference + normalize + validate),
and prints a one-line summary per crop. Writes a CSV summary.

GPU required (T4 on Colab). CPU fallback works but is slow.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import random
import time

from src import config
from src.tatr_postprocess import dedup_row_col_bands, normalize_tatr_prediction, validate_grid_geometry
from src.tatr_raw import RAW_BOX_KEYS, RAW_LABEL_TO_KEY


class _ReasonCollector:
    """Minimal logger shim: collects validate_grid_geometry failure reasons."""
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def log(self, sample_id: str, phase: str, error_type: str, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke: Phase 2 crops → TATR structure")
    p.add_argument("--crops-dir", type=Path, default=None,
                   help="directory of crop PNGs (default: config.LAYOUT_OUTPUT/crops)")
    p.add_argument("--n", type=int, default=5, help="number of crops to test")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for crop sampling (default: 42)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="TATR detection threshold")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="directory for smoke_structure.csv (default: config.LAYOUT_OUTPUT)")
    p.add_argument("--dedup-bands", action="store_true",
                   help="apply 1-D NMS to overlapping row/col bands before normalize")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    crops_dir = args.crops_dir or config.LAYOUT_OUTPUT / "crops"
    out_dir = args.out_dir or config.LAYOUT_OUTPUT

    all_crops = sorted(crops_dir.glob("*.png"))
    if not all_crops:
        print(f"[smoke] no crops found in {crops_dir}")
        return
    rng = random.Random(args.seed)
    crops = sorted(rng.sample(all_crops, min(args.n, len(all_crops))))
    print(f"[smoke] {len(crops)} crops sampled (seed={args.seed}) from {crops_dir}")

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
    csv_rows: list[dict] = []

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

        if args.dedup_bands:
            pred = dedup_row_col_bands(pred)
        canonical = normalize_tatr_prediction(pred)
        rows_sorted = sorted(pred["row_boxes"], key=lambda r: r["bbox"][1])
        cols_sorted = sorted(pred["col_boxes"], key=lambda c: c["bbox"][0])

        collector = _ReasonCollector()
        valid = validate_grid_geometry(
            rows_sorted, cols_sorted, canonical["cells"],
            logger=collector, sample_id=crop_path.stem,
        )

        status = "OK  " if valid else "WARN"
        if valid:
            passed += 1
        else:
            warned += 1

        reasons_str = "; ".join(collector.reasons) if collector.reasons else ""
        print(
            f"  {status}  {crop_path.name:<45}"
            f"  rows={canonical['num_rows']:>3}  cols={canonical['num_cols']:>2}"
            f"  cells={len(canonical['cells']):>4}  valid={valid}"
            + (f"  [{reasons_str}]" if reasons_str else "")
        )
        csv_rows.append({
            "crop": crop_path.name,
            "rows": canonical["num_rows"],
            "cols": canonical["num_cols"],
            "cells": len(canonical["cells"]),
            "valid": valid,
            "failure_reasons": reasons_str,
        })

    print(f"\n[smoke] {passed} OK / {warned} WARN out of {len(crops)}")
    if warned == 0:
        print("[smoke] structure handoff OK")

    csv_path = out_dir / "smoke_structure.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "rows", "cols", "cells", "valid", "failure_reasons"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[smoke] wrote {csv_path}")


if __name__ == "__main__":
    main()
