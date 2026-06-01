"""Visual preview of one FinTabNet.c sample (Phase 1B sanity check / deliverable #6).

Reads the GT structure XML + GT word tokens for one sample, builds the gt_filled
canonical table, and renders three things so a human can judge the reconstruction
without reading raw coordinates:
  1. <sid>_crop.png        - the real table crop
  2. <sid>_gt_grid.png     - the GT cell grid drawn over the crop (spanning emphasised)
  3. <sid>_gt_filled.html  - the reconstructed table with GT text filled in

This is gt_filled (text_source="gt"), used for QA validation only - never reported as
an extraction output (P4). Colab only (needs the crops); the rendering itself is the
unit-tested src/visualisation helpers.

Colab:  !python scripts/preview_table.py --sample-id AAL_2002_page_41_table_1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src import fintabnet_loader as fl  # noqa: E402
from src import visualisation as vis  # noqa: E402
from src.canonical_schema import TEXT_SOURCE_GT  # noqa: E402
from src.table_fill import fill_table  # noqa: E402


def _resolve_xml(root: Path, sample_id: str | None) -> Path:
    if sample_id:
        match = next(root.rglob(f"{sample_id}.xml"), None)
        if match is None:
            raise SystemExit(f"No XML found for sample_id {sample_id!r} under {root}")
        return match
    files = fl.find_xml_files(root, limit=1)
    if not files:
        raise SystemExit(f"No XML files under {root}")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-id", default=None,
                    help="stem of the sample; defaults to the first XML found")
    args = ap.parse_args()

    root = fl.download_structure()
    xml_path = _resolve_xml(root, args.sample_id)
    sample_id = xml_path.stem

    parsed = fl.parse_structure_xml(xml_path)
    words = fl.parse_words_json(fl.words_path_for(sample_id))
    table = fill_table(parsed, words, sample_id=sample_id, text_source=TEXT_SOURCE_GT)

    from PIL import Image
    crop = Image.open(fl.image_path_for(sample_id)).convert("RGB")

    out_dir = config.FIGURES / "phase1b_preview" / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    crop.save(out_dir / f"{sample_id}_crop.png")
    vis.draw_cell_grid(crop, table).save(out_dir / f"{sample_id}_gt_grid.png")
    (out_dir / f"{sample_id}_gt_filled.html").write_text(
        vis.topology_to_html(table), encoding="utf-8")

    # Which GT words did not land in any cell? assign_words_to_cells appends the same
    # word dicts into cell["words"], so identity tells us what was left out. This
    # classifies the coverage gap: out-of-table noise (footnotes, page numbers) vs real
    # cell content that the grid failed to capture.
    assigned_ids = {id(w) for c in table["cells"] for w in c.get("words", [])}
    unassigned = [w for w in words if id(w) not in assigned_ids]

    boxed = [c["bbox"] for c in table["cells"] if "bbox" in c]
    grid_bbox = [
        min(b[0] for b in boxed), min(b[1] for b in boxed),
        max(b[2] for b in boxed), max(b[3] for b in boxed),
    ] if boxed else None

    print(f"sample_id: {sample_id}")
    print(f"grid: {table['num_rows']} rows x {table['num_cols']} cols, "
          f"{len(table['cells'])} cells, {len(words)} GT words")
    print(f"grid bbox: {grid_bbox}")
    print(f"unassigned words: {len(unassigned)}")
    for w in unassigned[:30]:
        print(f"  {w['bbox']} {w['text']!r}")
    print(f"wrote preview to: {out_dir}")


if __name__ == "__main__":
    main()
