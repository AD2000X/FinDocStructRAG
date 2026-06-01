"""Inspect the real FinTabNet.c-Structure archive layout (first-run verification).

Downloads + extracts the structure archive, prints the directory tree, counts files,
and parses a few annotations so we can confirm the on-disk layout and class strings
before building the topology pipeline on top of them.

Colab:  !python scripts/inspect_fintabnet.py --limit 3
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import fintabnet_loader as fl  # noqa: E402


def list_repo_archives() -> None:
    """List every file in the HF dataset repo, so we can see which archive carries
    the GT cell text / word tokens (the structure XML holds only bboxes)."""
    from huggingface_hub import list_repo_files

    print(f"\n== Files in {fl.REPO_ID} (dataset repo) ==")
    for name in sorted(list_repo_files(fl.REPO_ID, repo_type="dataset")):
        print(f"  {name}")


def dump_json_schema(root: Path, limit: int = 1) -> None:
    """Dump the shape of any per-sample .json under the extraction root.

    Phase 1B needs a confirmed GT-text source before building gt_filled/. The structure
    XML carries no text, so we look for word/token JSON sidecars here.
    """
    json_files = sorted(root.rglob("*.json"))
    print(f"\n== .json files under structure root: {len(json_files)} ==")
    for jp in json_files[:limit]:
        print(f"\n--- {jp.relative_to(root)} ---")
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  could not parse: {exc}")
            continue
        if isinstance(data, list):
            print(f"  top-level: list of {len(data)} items")
            if data:
                print(f"  item[0] keys: {sorted(data[0].keys())}"
                      if isinstance(data[0], dict) else f"  item[0]: {data[0]!r}")
                print(f"  item[0] sample: {json.dumps(data[0])[:400]}")
        elif isinstance(data, dict):
            print(f"  top-level keys: {sorted(data.keys())}")
            print(f"  sample: {json.dumps(data)[:400]}")


def print_tree(root: Path, max_entries: int = 40) -> None:
    print(f"\n== Extraction root: {root} ==")
    entries = sorted(root.rglob("*"))
    print(f"Total entries under root: {len(entries)}")
    # Top-level layout.
    print("\nTop-level:")
    for p in sorted(root.iterdir()):
        kind = "dir " if p.is_dir() else "file"
        print(f"  [{kind}] {p.name}")
    # Extension histogram.
    ext_counts = Counter(p.suffix.lower() for p in entries if p.is_file())
    print("\nFile extensions:")
    for ext, n in ext_counts.most_common():
        print(f"  {ext or '(none)'}: {n}")
    # A sample of paths so we can see subfolder structure.
    print(f"\nFirst {max_entries} paths (relative):")
    for p in entries[:max_entries]:
        print(f"  {p.relative_to(root)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3, help="annotations to parse")
    ap.add_argument("--force", action="store_true", help="re-download/extract")
    ap.add_argument("--gt-text", action="store_true",
                    help="Phase 1B: list repo archives and dump any .json schema "
                         "to locate the GT cell-text source")
    args = ap.parse_args()

    if args.gt_text:
        list_repo_archives()

    root = fl.download_structure(force=args.force)
    print_tree(root)

    if args.gt_text:
        dump_json_schema(root)

    xml_files = fl.find_xml_files(root, limit=args.limit)
    print(f"\n== XML annotation files found: {len(fl.find_xml_files(root))} ==")
    if not xml_files:
        print("No .xml files found. The structure format differs from the assumption;")
        print("paste the 'Top-level' / paths above so the parser can be adjusted.")
        return

    agg = Counter()
    for xml_path in xml_files:
        parsed = fl.parse_structure_xml(xml_path)
        agg.update(parsed["class_counts"])
        print(f"\n--- {xml_path.name} ---")
        print(f"  image_filename: {parsed['image_filename']}")
        print(f"  class_counts:   {parsed['class_counts']}")
        print(f"  rows={len(parsed['row_boxes'])} cols={len(parsed['col_boxes'])} "
              f"spanning={len(parsed['spanning_cells'])} "
              f"col_headers={len(parsed['column_headers'])}")
        if parsed["row_boxes"]:
            print(f"  first row bbox: {parsed['row_boxes'][0]['bbox']}")

    print(f"\n== Aggregate class counts over {len(xml_files)} files ==")
    for name, n in agg.most_common():
        print(f"  {name!r}: {n}")


if __name__ == "__main__":
    main()
