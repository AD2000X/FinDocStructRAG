"""Phase 2 detector smoke: verify a DocLayNet layout detector BEFORE pinning it.

This is NOT Phase 2 pipeline code. It is a one-off gate (PLAN.md sec 3, Phase 2): confirm a
candidate HF object-detection model loads, exposes `config.id2label` with a Table class,
produces boxes on a page, and that a box converts to a pixel crop - on a real T4 so the
runtime / VRAM are measured, not guessed. It writes nothing to config; pinning
`config.LAYOUT_MODEL` happens only after a human reviews this output on T4.

The label set is already known from the model card / config.json (Aryn model: id 9 = "Table",
id 0 = "N/A"), so the smoke's real job is the parts that need the GPU and a real page:
load time, detections on an actual image, measured T4 VRAM, and the box -> crop step that
`bbox_utils` will later own.

Run on Colab T4 (the meaningful target):
    !pip install -q transformers timm
    !python scripts/smoke_layout_detector.py
Or locally on CPU as a load/shape check only (slower, NOT representative of T4 runtime).

Deps: transformers, timm (DETR-family backbone), pillow, requests, torch.
"""

from __future__ import annotations

import argparse
import time

DEFAULT_MODEL = "Aryn/deformable-detr-DocLayNet"
# The model card's bundled example page, so the smoke needs no DocLayNet download to run.
DEFAULT_IMAGE = (
    "https://huggingface.co/Aryn/deformable-detr-DocLayNet/resolve/main/examples/"
    "doclaynet_example_1.png"
)


def _load_image(src: str):
    """Open an image from a URL or a local path as RGB."""
    from PIL import Image

    if src.startswith(("http://", "https://")):
        import requests

        resp = requests.get(src, stream=True, timeout=30)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")
    return Image.open(src).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 layout-detector smoke (writes no config).")
    ap.add_argument("--model-id", default=DEFAULT_MODEL, help=f"HF id (default: {DEFAULT_MODEL})")
    ap.add_argument(
        "--image",
        action="append",
        help="image URL or local path; repeatable. Default: the model card example page.",
    )
    ap.add_argument("--threshold", type=float, default=0.7, help="score threshold (default: 0.7)")
    ap.add_argument(
        "--save-crop",
        metavar="PATH",
        default=None,
        help="optional: save the top-score detection crop here (PNG) to eyeball it",
    )
    args = ap.parse_args()
    images = args.image or [DEFAULT_IMAGE]

    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device={device}  cuda_available={torch.cuda.is_available()}")
    if device == "cpu":
        print("WARNING: no CUDA - this is a load/shape check only, NOT a T4 runtime measurement.")

    # 1. Load processor + model.
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForObjectDetection.from_pretrained(args.model_id).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[load] {args.model_id} on {device} in {time.time() - t0:.1f}s, {n_params:.1f}M params")

    # 2. id2label must exist and contain a Table class (drives LAYOUT_LABEL_MAP later).
    id2label = model.config.id2label
    print(f"[id2label] {id2label}")
    table_ids = [i for i, lab in id2label.items() if "table" in str(lab).lower()]
    assert table_ids, f"no Table class in id2label - wrong model for table crops: {id2label}"
    print(f"[id2label] Table class present at id(s) {table_ids}")
    table_id_set = {int(i) for i in table_ids}

    # 3. Inference + post-process per image; measure latency and (on cuda) peak VRAM.
    any_table_box = False
    for src in images:
        image = _load_image(src)
        width, height = image.size
        inputs = processor(images=image, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        dt = time.time() - t1

        target_sizes = torch.tensor([(height, width)])  # (h, w), CPU is fine for post-process
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=args.threshold
        )[0]
        n = len(results["scores"])
        print(f"\n[detect] {src}  size=({width}, {height})  {n} boxes >= {args.threshold} in {dt:.2f}s")
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xyxy = [round(v, 1) for v in box.tolist()]
            print(f"    {id2label[label.item()]:<16} {score.item():.3f}  {xyxy}")
        if device == "cuda":
            print(f"[vram] peak {torch.cuda.max_memory_allocated() / 1e6:.0f} MB on this image")

        # 4. Table-crop gate: crop the top-score TABLE box (the actual Phase 2 carry-forward),
        #    not any top-score label, so a page of only Text/Title does not pass as OK.
        table_idx = [k for k, lab in enumerate(results["labels"].tolist()) if lab in table_id_set]
        if table_idx:
            any_table_box = True
            best = max(table_idx, key=lambda k: results["scores"][k].item())
            x1, y1, x2, y2 = (int(round(v)) for v in results["boxes"][best].tolist())
            box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))  # clamp to page
            crop = image.crop(box)  # PIL (left, upper, right, lower) == xyxy
            print(
                f"[crop] top Table box (score {results['scores'][best].item():.3f}) {box} "
                f"-> crop size {crop.size}"
            )
            if args.save_crop:
                crop.save(args.save_crop)
                print(f"[crop] saved -> {args.save_crop}")
        else:
            print(f"[crop] no Table box >= {args.threshold} on this image - no table crop to verify here")

    # Fail-fast gate: the carry-forward is the Table crop, so a run with no Table box is a FAIL,
    # even if the model loaded and detected other classes.
    assert any_table_box, (
        f"no Table box detected >= {args.threshold} across {len(images)} image(s) - the table-crop "
        "handoff is NOT verified; lower --threshold or check the model/page before pinning"
    )
    print(
        "\nsmoke OK - model loads, id2label has Table, a Table box was detected and cropped. "
        "Pin config.LAYOUT_MODEL only after reviewing this output on T4."
    )


if __name__ == "__main__":
    main()
