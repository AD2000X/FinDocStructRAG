"""Phase 1A deliverable figures (DESIGN_SPEC 5.14). CPU-only, run on Colab.

Reads a run's persisted artifacts (tatr_raw/, tatr_predicted/, the evaluation report) and
the table crops from the dataset cache, and writes the deliverable figures under
outputs/figures/phase1a/. No GPU. No GT overlay (P4: these are prediction figures only;
#6 GT-filled HTML is a Phase 1B deliverable and is intentionally not produced here).

    python scripts/render_phase1a_figures.py --run-id mvp_rand

By default it picks one clean sample (no geometry flags) and one failure sample (#8, has
geometry flags); override with --sample-id / --failure-sample-id. The drawing/HTML logic
is unit-tested in tests/test_visualisation.py; this script is the glue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src import visualisation as vis  # noqa: E402
from src.fintabnet_loader import structure_root  # noqa: E402
from src.run_manifest import read_completed  # noqa: E402

PHASE = "phase1a"


def _image_index(root) -> dict:
    return {p.name: p for p in root.rglob("*.jpg")}


def _load_raw(sample_id: str):
    p = config.TABLES_TATR_RAW / f"{sample_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _render_sample(raw, pred_table, image_path, out_dir: Path) -> list[str]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    written = []

    def save_png(name, im):
        im.save(out_dir / name)
        written.append(name)

    save_png("01_crop.png", img)
    save_png("02_tatr_overlay.png", vis.draw_tatr_overlay(img, raw))
    save_png("03_cell_grid.png", vis.draw_cell_grid(img, pred_table))
    if any(vis.is_spanning(c) for c in pred_table.get("cells", [])):
        save_png("04_spanning.png", vis.draw_spanning_cells(img, pred_table))

    (out_dir / "05_geometry.txt").write_text(
        vis.geometry_report(raw), encoding="utf-8")
    written.append("05_geometry.txt")
    (out_dir / "07_predicted_table.html").write_text(
        vis.topology_to_html(pred_table), encoding="utf-8")
    written.append("07_predicted_table.html")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="mvp_rand")
    ap.add_argument("--sample-id", default=None, help="override the clean sample")
    ap.add_argument("--failure-sample-id", default=None, help="override the #8 sample")
    args = ap.parse_args()

    rows = read_completed(config.MANIFESTS / f"{PHASE}_{args.run_id}.csv")
    if not rows:
        raise SystemExit(f"no completed samples for run-id {args.run_id}")
    by_id = {r["sample_id"]: r for r in rows}

    images = _image_index(structure_root())
    fig_root = config.FIGURES / PHASE
    fig_root.mkdir(parents=True, exist_ok=True)

    def resolve(sample_id, want_failure):
        """Return (sid, raw, pred_table, image_path) for an override id, or the first
        sample matching the wanted failure-ness when no id is given."""
        candidates = [sample_id] if sample_id else list(by_id)
        for sid in candidates:
            if sid not in by_id:
                continue
            raw = _load_raw(sid)
            if raw is None:
                continue
            if sample_id is None and vis.is_failure_candidate(raw) != want_failure:
                continue
            pred_path = Path(by_id[sid]["output_path"])
            image_path = images.get(raw.get("image_filename"))
            if not pred_path.exists() or image_path is None:
                continue
            pred_table = json.loads(pred_path.read_text(encoding="utf-8"))
            return sid, raw, pred_table, image_path
        return None

    # #9 metrics summary (with the subset disclaimer).
    summary_path = config.EVALUATION / f"{PHASE}_topology_{args.run_id}.json"
    note = (f"run-id={args.run_id}; fixed subset; "
            f"metrics over successful samples ({len(rows)})")
    summary_out = fig_root / "09_metrics_summary.html"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_out.write_text(vis.summary_to_html(summary, note), encoding="utf-8")

    for tag, picked in (
        ("clean", resolve(args.sample_id, want_failure=False)),
        ("failure", resolve(args.failure_sample_id, want_failure=True)),
    ):
        if picked is None:
            print(f"{tag}: none found")
            continue
        sid, raw, pred_table, image_path = picked
        out_dir = fig_root / f"{tag}_{sid}"
        written = _render_sample(raw, pred_table, image_path, out_dir)
        print(f"{tag}: {sid} -> {out_dir}")
        for name in written:
            print(f"    {name}")

    print(f"metrics summary -> {summary_out}")


if __name__ == "__main__":
    main()
