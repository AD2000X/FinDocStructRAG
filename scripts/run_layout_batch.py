#!/usr/bin/env python3
"""Phase 2 layout batch runner: detect + crop a fixed DocLayNet subset.

Runs detect_layout (primary Aryn + optional TATR fallback) over a seed-sampled
subset of DocLayNet val pages. Writes per-page artifacts and a manifest.

Outputs under <out_dir> (default: config.LAYOUT_OUTPUT):
  regions/<page_id>.json          - detected Region list
  crops/<page_id>_table_<i>.png   - one PNG per cropped table (score >= table_threshold)
  manifest.csv                    - one row per page

Manifest columns:
  page_id, status (processed/failed), num_regions, num_tables,
  num_cropped, fallback_used, error

Fixed seeds (PLAN §0): debug=7 (n=20), mvp=42 (n=200).
Not tested in ordinary pytest (requires DocLayNet download + GPU).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 layout batch runner")
    p.add_argument("--split", default="val")
    p.add_argument("--seed", type=int, default=7,
                   help="random seed for page sampling (debug=7, mvp=42)")
    p.add_argument("--n", type=int, default=20, help="number of pages to process")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output root (default: config.LAYOUT_OUTPUT)")
    p.add_argument("--primary-threshold", type=float, default=0.3,
                   help="score cutoff inside build_layout_detector")
    p.add_argument("--table-threshold", type=float, default=0.5,
                   help="score threshold for: fallback detector, fallback trigger, crop filter")
    p.add_argument("--dedup-iou", type=float, default=0.5)
    p.add_argument("--no-fallback", action="store_true",
                   help="disable TATR fallback (primary only)")
    return p.parse_args()


def _print_summary(manifest_path: Path) -> None:
    rows = list(csv.DictReader(manifest_path.open()))
    status_counts: dict[str, int] = {}
    total_cropped = fallback_count = 0
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        total_cropped += int(r["num_cropped"])
        if r["fallback_used"] == "True":
            fallback_count += 1
    print(f"  status      : {status_counts}")
    print(f"  tables cropped : {total_cropped}")
    print(f"  fallback used  : {fallback_count} / {len(rows)}")


def main() -> None:
    args = parse_args()

    from src import config
    from src.bbox_utils import crop_with_padding
    from src.layout_detector import build_layout_detector
    from src.layout_parsing import TABLE_LABEL, detect_layout
    from src.table_detection import build_table_transformer_detector

    out_dir: Path = args.out_dir or config.LAYOUT_OUTPUT
    regions_dir = out_dir / "regions"
    crops_dir = out_dir / "crops"
    regions_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("[batch] loading detectors ...")
    t0 = time.time()
    layout_det = build_layout_detector(
        config.LAYOUT_MODEL, threshold=args.primary_threshold
    )
    table_det = (
        None
        if args.no_fallback
        else build_table_transformer_detector(
            config.TATR_DETECTION_MODEL, threshold=args.table_threshold
        )
    )
    print(f"[batch] detectors ready in {time.time() - t0:.1f}s")

    print(f"[batch] loading DocLayNet {args.split} (seed={args.seed} n={args.n}) ...")
    from datasets import load_dataset

    ds = load_dataset("docling-project/DocLayNet-v1.1", split=args.split)
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(ds)), k=min(args.n, len(ds))))
    subset = ds.select(indices)
    print(f"[batch] {len(indices)} pages selected from {len(ds)} total")

    _FIELDS = [
        "page_id", "status",
        "num_regions", "num_tables", "num_cropped",
        "fallback_used", "error",
    ]
    manifest_path = out_dir / "manifest.csv"

    with manifest_path.open("w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=_FIELDS)
        writer.writeheader()

        for i, (orig_idx, ex) in enumerate(zip(indices, subset)):
            page_id = f"val_{orig_idx:06d}"
            print(f"[{i + 1:3d}/{len(indices)}] {page_id}", end="  ", flush=True)
            row: dict = {
                "page_id": page_id, "status": "failed",
                "num_regions": 0, "num_tables": 0, "num_cropped": 0,
                "fallback_used": False, "error": "",
            }
            try:
                img = ex["image"].convert("RGB")
                t1 = time.time()
                regions = detect_layout(
                    img, layout_det, table_det,
                    min_table_score=args.table_threshold,
                    dedup_iou=args.dedup_iou,
                )
                elapsed = time.time() - t1

                fallback_used = any(r.source == "table_fallback" for r in regions)
                all_tables = [r for r in regions if r.label == TABLE_LABEL]
                crop_tables = [r for r in all_tables if r.score >= args.table_threshold]

                (regions_dir / f"{page_id}.json").write_text(
                    json.dumps(
                        [
                            {
                                "label": r.label,
                                "score": round(r.score, 4),
                                "box": [round(c, 1) for c in r.box],
                                "source": r.source,
                            }
                            for r in regions
                        ],
                        indent=2,
                    )
                )

                num_cropped = 0
                for j, r in enumerate(crop_tables):
                    try:
                        crop = crop_with_padding(img, r.box, pad=4)
                        crop.save(crops_dir / f"{page_id}_table_{j}.png")
                        num_cropped += 1
                    except ValueError as e:
                        print(f"\n    [warn] {page_id} crop {j} degenerate: {e}")

                row.update({
                    "status": "processed",
                    "num_regions": len(regions),
                    "num_tables": len(all_tables),
                    "num_cropped": num_cropped,
                    "fallback_used": fallback_used,
                })
                print(
                    f"regions={len(regions):2d}  tables={len(all_tables)}"
                    f"  cropped={num_cropped}  fallback={fallback_used}"
                    f"  {elapsed:.2f}s"
                )

            except Exception:
                err = traceback.format_exc().splitlines()[-1]
                row["error"] = err
                print(f"FAILED: {err}")
                traceback.print_exc(file=sys.stderr)

            writer.writerow(row)
            mf.flush()

    print(f"\n[batch] done  manifest -> {manifest_path}")
    _print_summary(manifest_path)


if __name__ == "__main__":
    main()
