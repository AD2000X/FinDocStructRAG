#!/usr/bin/env python3
"""Phase 2 IoU diagnostic for fixed DocLayNet subsets.

Answers three questions about the current layout-crop policy:
  Q1: Did fallback fire because primary missed tables, or only because scores were low?
  Q2: When fallback fires, is fallback IoU actually better than primary?
  Q3: How sensitive are crop results to table_threshold?

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
from src.layout_parsing import (
    DEFAULT_TABLE_DEDUP_IOU,
    DEFAULT_TABLE_SCORE,
    TABLE_LABEL,
    Region,
)


class _Row(NamedTuple):
    page_id: str
    gt_tables: int
    # primary-alone (before fallback logic)
    primary_tables: int
    primary_max_score: float
    # fallback-alone
    fallback_tables: int
    # detect_layout merged result: candidates (all tables) vs cropped (score >= table_threshold)
    num_candidate_tables: int
    num_crop_tables: int
    fallback_used: bool
    # IoU vs GT: candidate = all detected, crop = score-filtered (matches batch runner)
    best_iou_primary: float
    best_iou_fallback: float
    best_iou_candidate: float
    best_iou_crop: float
    # table-level greedy matching (crops vs GT)
    matched_50: int
    matched_75: int


def _best_iou(regions: list[Region], gt_boxes: list) -> float:
    if not regions or not gt_boxes:
        return 0.0
    return max(iou(r.box, g) for r in regions for g in gt_boxes)


def _greedy_match(pred_boxes: list, gt_boxes: list, threshold: float) -> int:
    """Count GT tables matched at IoU >= threshold by greedy assignment (highest IoU first)."""
    if not pred_boxes or not gt_boxes:
        return 0
    pairs = sorted(
        ((iou(p, g), pi, gi) for pi, p in enumerate(pred_boxes) for gi, g in enumerate(gt_boxes)),
        reverse=True,
    )
    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    for v, pi, gi in pairs:
        if v < threshold:
            break
        if pi not in matched_preds and gi not in matched_gts:
            matched_preds.add(pi)
            matched_gts.add(gi)
    return len(matched_gts)


def _gt_table_boxes(ex: dict, img_w: int, img_h: int) -> list[tuple]:
    """Extract GT table boxes (xyxy pixel) from a DocLayNet dataset example.

    DocLayNet uses COCO-format annotations: bbox is [x, y, w, h] in pixel
    coordinates for 1025x1025 pages. category_id 9 == Table (verified in smoke).
    Key may be 'bboxes' or 'bbox' depending on HF dataset version.
    """
    bboxes = ex.get("bboxes", ex.get("bbox", []))
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
    p.add_argument("--primary-threshold", type=float, default=DEFAULT_TABLE_SCORE)
    p.add_argument("--table-threshold", type=float, default=DEFAULT_TABLE_SCORE,
                   help="active threshold used for fallback trigger and final crop")
    p.add_argument("--dedup-iou", type=float, default=DEFAULT_TABLE_DEDUP_IOU)
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
        if i == 0:
            print(f"[diag] example keys: {list(ex.keys())}")
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
        final_crop_tables = [r for r in final_tables if r.score >= args.table_threshold]
        fallback_used = any(r.source == "table_fallback" for r in final_regions)
        crop_boxes = [r.box for r in final_crop_tables]
        matched_50 = _greedy_match(crop_boxes, gt_boxes, 0.50)
        matched_75 = _greedy_match(crop_boxes, gt_boxes, 0.75)

        row = _Row(
            page_id=page_id,
            gt_tables=len(gt_boxes),
            primary_tables=len(primary_tables),
            primary_max_score=round(primary_max_score, 4),
            fallback_tables=len(fallback_tables),
            num_candidate_tables=len(final_tables),
            num_crop_tables=len(final_crop_tables),
            fallback_used=fallback_used,
            best_iou_primary=round(_best_iou(primary_tables, gt_boxes), 4),
            best_iou_fallback=round(_best_iou(fallback_tables, gt_boxes), 4),
            best_iou_candidate=round(_best_iou(final_tables, gt_boxes), 4),
            best_iou_crop=round(_best_iou(final_crop_tables, gt_boxes), 4),
            matched_50=matched_50,
            matched_75=matched_75,
        )
        rows.append(row)
        print(
            f"gt={row.gt_tables}"
            f"  prim={row.primary_tables}(max={row.primary_max_score:.2f})"
            f"  fb={row.fallback_tables}"
            f"  iou p/fb/cand/crop={row.best_iou_primary:.2f}/{row.best_iou_fallback:.2f}/{row.best_iou_candidate:.2f}/{row.best_iou_crop:.2f}"
            f"  m50={row.matched_50}/{row.gt_tables}"
            f"  fb_used={row.fallback_used}  {elapsed:.2f}s"
        )

    # Write CSV with a mode suffix so positive/negative runs do not overwrite each other.
    mode_suffix = "_pos" if args.require_table_gt else ("_neg" if args.exclude_table_gt else "")
    diag_path = out_dir / f"diagnostic{mode_suffix}.csv"
    with diag_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_Row._fields))
        w.writeheader()
        for r in rows:
            w.writerow(r._asdict())
    print(f"\n[diag] wrote {diag_path}")

    # Summary
    has_gt = [r for r in rows if r.gt_tables > 0]
    fb_pages = [r for r in has_gt if r.fallback_used]
    print(f"\n{'='*60}")
    print(f"Pages with GT tables : {len(has_gt)} / {len(rows)}")
    print(f"Fallback used        : {len(fb_pages)} / {len(has_gt)} (GT-table pages only)")
    print(f"\n  mean best_iou_primary   : {_mean([r.best_iou_primary   for r in has_gt]):.3f}")
    print(f"  mean best_iou_fallback  : {_mean([r.best_iou_fallback  for r in has_gt]):.3f}")
    print(f"  mean best_iou_candidate : {_mean([r.best_iou_candidate for r in has_gt]):.3f}  (all detected tables)")
    print(f"  mean best_iou_crop      : {_mean([r.best_iou_crop      for r in has_gt]):.3f}  (score >= table_threshold)")

    # Q1: primary miss vs low score
    print(f"\n-- Q1: on {len(fb_pages)} fallback pages --")
    found = [r for r in fb_pages if r.primary_tables > 0]
    missed = [r for r in fb_pages if r.primary_tables == 0]
    print(f"  primary found table (but < thresh) : {len(found)}")
    if found:
        lo = min(r.primary_max_score for r in found)
        hi = max(r.primary_max_score for r in found)
        print(f"    score range  : {lo:.2f} - {hi:.2f}")
        print(f"    mean primary IoU on these : {_mean([r.best_iou_primary for r in found]):.3f}")
    print(f"  primary completely missed table    : {len(missed)}")

    # Q2: fallback IoU vs primary IoU on fallback pages
    if fb_pages:
        print(f"\n-- Q2: fallback vs primary IoU (on {len(fb_pages)} fallback pages) --")
        fb_better = [r for r in fb_pages if r.best_iou_fallback > r.best_iou_primary]
        prim_better = [r for r in fb_pages if r.best_iou_primary > r.best_iou_fallback]
        equal = [r for r in fb_pages if r.best_iou_fallback == r.best_iou_primary]
        print(f"  fallback better  : {len(fb_better)}")
        print(f"  primary better   : {len(prim_better)}")
        print(f"  equal (both 0?)  : {len(equal)}")
        print(f"  mean iou_fallback : {_mean([r.best_iou_fallback for r in fb_pages]):.3f}")
        print(f"  mean iou_primary  : {_mean([r.best_iou_primary  for r in fb_pages]):.3f}")

    # Q3: threshold sensitivity (simulate different table_threshold values)
    if has_gt:
        print(f"\n-- Q3: threshold sensitivity (simulated, {len(has_gt)} GT-table pages) --")
        print(f"  Rule: fallback fires only when primary_tables >= 1 and score < thresh.")
        print(f"  Note: IoU values are pre-dedup proxies (val_005241-style collapses not captured).")
        print(f"  {'thresh':>7}  {'fb_pages':>8}  {'iou_crop_sim(pre-dedup)':>22}")
        for thresh in [0.30, 0.40, 0.50, 0.60, 0.70]:
            # Only pages where primary found >= 1 table but none above thresh trigger fallback
            sim_fb = sum(
                1 for r in has_gt if r.primary_tables > 0 and r.primary_max_score < thresh
            )
            sim_ious = []
            for r in has_gt:
                if r.primary_max_score >= thresh:
                    sim_ious.append(r.best_iou_primary)
                elif r.primary_tables > 0:
                    # fallback fires: use fallback IoU as proxy
                    sim_ious.append(r.best_iou_fallback)
                else:
                    # Primary found zero tables: fallback skipped, no crop.
                    sim_ious.append(0.0)
            print(f"  {thresh:>7.2f}  {sim_fb:>8}  {_mean(sim_ious):>22.3f}")

    # Table-level matching summary (GT-table pages only)
    if has_gt:
        gt_total = sum(r.gt_tables for r in has_gt)
        pred_total = sum(r.num_crop_tables for r in has_gt)
        m50 = sum(r.matched_50 for r in has_gt)
        m75 = sum(r.matched_75 for r in has_gt)
        prec50 = f"{m50 / pred_total:.3f}" if pred_total else "N/A"
        prec75 = f"{m75 / pred_total:.3f}" if pred_total else "N/A"
        print(f"\n-- Table-level matching ({len(has_gt)} GT-table pages) --")
        print(f"  GT tables total    : {gt_total}")
        print(f"  crops total        : {pred_total}")
        print(f"  matched@0.50       : {m50}   recall={m50 / gt_total:.3f}  precision={prec50}")
        print(f"  matched@0.75       : {m75}   recall={m75 / gt_total:.3f}  precision={prec75}")
        print(f"  missed GT tables   : {gt_total - m50}  (no crop with IoU >= 0.50)")
        print(f"  extra crops        : {pred_total - m50}  (crops not matching any GT at IoU >= 0.50)")

    # False-positive report: only printed when all pages have no GT table
    no_gt = [r for r in rows if r.gt_tables == 0]
    if no_gt and len(no_gt) == len(rows):
        fp_primary = sum(1 for r in no_gt if r.primary_tables > 0)
        fp_fallback = sum(1 for r in no_gt if r.fallback_used)
        fp_crop = sum(1 for r in no_gt if r.num_crop_tables > 0)
        print(f"\n-- False-positive rate ({len(no_gt)} table-free pages) --")
        print(f"  primary detected table   : {fp_primary} / {len(no_gt)}")
        print(f"  fallback triggered       : {fp_fallback} / {len(no_gt)}")
        print(f"  final crop produced      : {fp_crop} / {len(no_gt)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
