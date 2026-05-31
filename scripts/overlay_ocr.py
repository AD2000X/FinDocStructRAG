"""Phase 1B spatial debug: GT cells + TATR pred cells + OCR word boxes on a crop (CPU).

For one sample, overlays the GT cell grid (green), the TATR predicted cell grid (gray),
and the OCR word/detection boxes (red, read from ocr_filled) on the table crop. This
shows whether OCR detection boxes span GT column boundaries (a column-grouping error) or
the predicted columns are shifted - the IP/MA-style failures where adjacent numeric
columns merge into one cell. Reads persisted tables + the crop; runs no model.

    python scripts/overlay_ocr.py --sample-id IP_2012_page_114_table_2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src import visualisation as vis  # noqa: E402
from src.fintabnet_loader import download_structure, image_path_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-id", required=True, help="sample stem to overlay")
    args = ap.parse_args()

    sample_id = args.sample_id
    ocr_path = config.TABLES_OCR_FILLED / f"{sample_id}.json"
    gt_path = config.TABLES_GT_FILLED / f"{sample_id}.json"
    if not ocr_path.exists():
        raise SystemExit(f"no ocr_filled for {sample_id}: {ocr_path}")
    if not gt_path.exists():
        raise SystemExit(f"no gt_filled for {sample_id}: {gt_path}")

    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    gt = json.loads(gt_path.read_text(encoding="utf-8"))

    download_structure()
    from PIL import Image
    crop = Image.open(image_path_for(sample_id)).convert("RGB")

    out_dir = config.FIGURES / "phase1b_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}_ocr_debug.png"
    vis.draw_ocr_debug_overlay(crop, gt, ocr).save(out_path)

    n_words = sum(len(c.get("words", [])) for c in ocr["cells"])
    print(f"sample_id: {sample_id}")
    print(f"GT cells: {len(gt['cells'])}  pred cells: {len(ocr['cells'])}  "
          f"OCR word boxes: {n_words}")
    print(f"wrote overlay to: {out_path}")


if __name__ == "__main__":
    main()
