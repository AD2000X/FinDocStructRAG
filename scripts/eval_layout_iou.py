#!/usr/bin/env python3
"""Phase 2 IoU diagnostic for the fixed debug subset (default seed=7, n=20).

Answers three questions about the batch runner's 10/20 fallback rate:
  Q1: Did primary miss the table entirely, or just score < threshold?
  Q2: Is fallback IoU actually better than primary?
  Q3: Is table_threshold=0.5 too strict?

Strategy: re-run detection capturing primary-alone and fallback-alone results
BEFORE the detect_layout merge/dedup, so primary IoU is not contaminated by
dedup removal. The 'final' IoU is from the actual detect_layout output.

Outputs:
  <out_dir>/diagnostic.csv  - one row per page
  printed summary per question + threshold sensitivity table

Not tested in ordinary pytest (requires DocLayNet download + GPU).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import random
import time
from typing import NamedTuple

from src.bbox_utils import iou, xywh_to_xyxy
from src.layout_parsing import TABLE_LABEL, Region


class _Row(NamedTuple):
    page_id: str
    gt_tables: int
    # primary-alone (before fallback logic)
    primary_tables: int
    primary_max_score: float
    # fallback-alone
    fallback_tables: int
    # detect_layout merged result
    final_tables: int
    fallback_used: bool
    # IoU vs GT
    best_iou_primary: float
    best_iou_fallback: float
    best_iou_final: float


def _best_iou(regions: list[Region], gt_boxes: list) -> float:
    if not regions or not gt_boxes:
        return 0.0
    return max(iou(r.box, g) for r in regions for g in gt_boxes)


def _gt_table_boxes(ex: dict, img_w: int, img_h: int) -> list[tuple]:
    """Extract GT table boxes (xyxy pixel) from a DocLayNet dataset example.

    DocLayNet uses COCO-format annotations: bbox is [x, y, w, h] in pixel
    coordinates for 1025x1025 pages. category_id 9 == Table (verified in smoke).
    """
    bboxes = ex.get("bbox", [])
    cats = ex.get("category_id", [])
    boxes = []
    for cat, bbox in zip(cats, bboxes):
        if cat != 9:
            continue
        x, y, w, h = bbox
        # Guard: if coords look normalized (all <= 2.0), scale to pixels
        if max(abs(x), abs(y), abs(w), abs(h)) <= 2.0:
            x, y, w, h = x * img_w, y * img_h, w * img_w, h * img_h
        boxes.append(xywh_to_xyxy((x, y, w, h)))
    return boxes


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 IoU diagnostic")
    p.add_argument("--split", default="val")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--primary-threshold", type=float, default=0.3)
    p.add_argument("--table-threshold", type=float, default=0.5,
                   help="active threshold used for fallback trigger and final crop")
    p.add_argument("--dedup-iou", type=float, default=0.5)
    p.add_argument("--require-table-gt", action="store_true",
                   help="only sample pages with GT Table annotations (category_id 9)")
    p.add_argument("--exclude-table-gt", action="store_true",
                   help="only sample pages with no GT Table (false-positive diagnostic)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src import config
    from src.layout_detector import build_layout_detector
    from src.layout_parsing import detect_layout
    from src.table_detection import build_table_transformer_detector

    out_dir: Path = args.out_dir or config.LAYOUT_OUTPUT

    print("[diag] loading detectors ...")
    t0 = time.time()
    layout_det = build_layout_detector(config.LAYOUT_MODEL, threshold=args.primary_threshold)
    table_det = build_table_transformer_detector(
        config.TATR_DETECTION_MODEL, threshold=args.table_threshold
    )
    print(f"[diag] detectors ready in {time.time() - t0:.1f}s")

    from datasets import load_dataset

    ds = load_dataset("docling-project/DocLayNet-v1.1", split=args.split)
    rng = random.Random(args.seed)
    if args.require_table_gt:
        all_cats = ds["category_id"]
        pool = [i for i, cats in enumerate(all_cats) if 9 in cats]
    elif args.exclude_table_gt:
        all_cats = ds["category_id"]
        pool = [i for i, cats in enumerate(all_cats) if 9 not in cats]
    else:
        pool = list(range(len(ds)))
    indices = sorted(rng.sample(pool, k=min(args.n, len(pool))))
    subset = ds.select(indices)
    mode = "table-only" if args.require_table_gt else ("no-table" if args.exclude_table_gt else "random")
    print(f"[diag] {len(indices)} pages (seed={args.seed} split={args.split} mode={mode} pool={len(pool)})")

    rows: list[_Row] = []

    for i, (orig_idx, ex) in enumerate(zip(indices, subset)):
        page_id = f"val_{orig_idx:06d}"
        print(f"[{i + 1:3d}/{len(indices)}] {page_id}", end="  ", flush=True)

        img = ex["image"].convert("RGB")
        iw, ih = img.size
        gt_boxes = _gt_table_boxes(ex, iw, ih)

        # Primary alone (direct call, no fallback logic)
        primary_all = layout_det(img)
        primary_tables = [r for r in primary_all if r.label == TABLE_LABEL]
        primary_max_score = max((r.score for r in primary_tables), default=0.0)

        # Fallback alone (direct call)
        fallback_all = table_det(img)
        fallback_tables = [r for r in fallback_all if r.label == TABLE_LABEL]

        # detect_layout: sequential + dedup (the actual pipeline result)
        t1 = time.time()
        final_regions = detect_layout(
            img, layout_det, table_det,
            min_table_score=args.table_threshold,
            dedup_iou=args.dedup_iou,
        )
        elapsed = time.time() - t1
        final_tables = [r for r in final_regions if r.label == TABLE_LABEL]
        fallback_used = any(r.source == "table_fallback" for r in final_regions)

        row = _Row(
            page_id=page_id,
            gt_tables=len(gt_boxes),
            primary_tables=len(primary_tables),
            primary_max_score=round(primary_max_score, 4),
            fallback_tables=len(fallback_tables),
            final_tables=len(final_tables),
            fallback_used=fallback_used,
            best_iou_primary=round(_best_iou(primary_tables, gt_boxes), 4),
            best_iou_fallback=round(_best_iou(fallback_tables, gt_boxes), 4),
            best_iou_final=round(_best_iou(final_tables, gt_boxes), 4),
        )
        rows.append(row)
        print(
            f"gt={row.gt_tables}"
            f"  prim={row.primary_tables}(max={row.primary_max_score:.2f})"
            f"  fb={row.fallback_tables}"
            f"  iou p/fb/fin={row.best_iou_primary:.2f}/{row.best_iou_fallback:.2f}/{row.best_iou_final:.2f}"
            f"  fb_used={row.fallback_used}  {elapsed:.2f}s"
        )

    # Write CSV
    diag_path = out_dir / "diagnostic.csv"
    with diag_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_Row._fields))
        w.writeheader()
        for r in rows:
            w.writerow(r._asdict())
    print(f"\n[diag] wrote {diag_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    has_gt = [r for r in rows if r.gt_tables > 0]
    fb_pages = [r for r in has_gt if r.fallback_used]
    print(f"\n{'='*60}")
    print(f"Pages with GT tables : {len(has_gt)} / {len(rows)}")
    print(f"Fallback used        : {len(fb_pages)} / {len(has_gt)} (GT-table pages only)")
    print(f"\n  mean best_iou_primary  : {_mean([r.best_iou_primary  for r in has_gt]):.3f}")
    print(f"  mean best_iou_fallback : {_mean([r.best_iou_fallback for r in has_gt]):.3f}")
    print(f"  mean best_iou_final    : {_mean([r.best_iou_final    for r in has_gt]):.3f}")

    # Q1 ─ primary miss vs low score
    print(f"\n── Q1: on {len(fb_pages)} fallback pages ──")
    found = [r for r in fb_pages if r.primary_tables > 0]
    missed = [r for r in fb_pages if r.primary_tables == 0]
    print(f"  primary found table (but < thresh) : {len(found)}")
    if found:
        lo = min(r.primary_max_score for r in found)
        hi = max(r.primary_max_score for r in found)
        print(f"    score range  : {lo:.2f} – {hi:.2f}")
        print(f"    mean primary IoU on these : {_mean([r.best_iou_primary for r in found]):.3f}")
    print(f"  primary completely missed table    : {len(missed)}")

    # Q2 ─ fallback IoU vs primary IoU on fallback pages
    if fb_pages:
        print(f"\n── Q2: fallback vs primary IoU (on {len(fb_pages)} fallback pages) ──")
        fb_better = [r for r in fb_pages if r.best_iou_fallback > r.best_iou_primary]
        prim_better = [r for r in fb_pages if r.best_iou_primary > r.best_iou_fallback]
        equal = [r for r in fb_pages if r.best_iou_fallback == r.best_iou_primary]
        print(f"  fallback better  : {len(fb_better)}")
        print(f"  primary better   : {len(prim_better)}")
        print(f"  equal (both 0?)  : {len(equal)}")
        print(f"  mean iou_fallback : {_mean([r.best_iou_fallback for r in fb_pages]):.3f}")
        print(f"  mean iou_primary  : {_mean([r.best_iou_primary  for r in fb_pages]):.3f}")

    # Q3 ─ threshold sensitivity (simulate different table_threshold values)
    if has_gt:
        print(f"\n── Q3: threshold sensitivity (simulated, {len(has_gt)} GT-table pages) ──")
        print(f"  {'thresh':>7}  {'fb_pages':>8}  {'mean_iou_final':>14}")
        for thresh in [0.30, 0.40, 0.50, 0.60, 0.70]:
            sim_fb = sum(1 for r in has_gt if r.primary_max_score < thresh)
            # If primary above thresh -> use primary IoU; else use fallback IoU
            sim_ious = [
                r.best_iou_primary if r.primary_max_score >= thresh else r.best_iou_fallback
                for r in has_gt
            ]
            print(f"  {thresh:>7.2f}  {sim_fb:>8}  {_mean(sim_ious):>14.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
