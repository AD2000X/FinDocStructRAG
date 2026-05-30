"""Inspect the real FinTabNet.c-Structure archive layout (first-run verification).

Downloads + extracts the structure archive, prints the directory tree, counts files,
and parses a few annotations so we can confirm the on-disk layout and class strings
before building the topology pipeline on top of them.

Colab:  !python scripts/inspect_fintabnet.py --limit 3
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import fintabnet_loader as fl  # noqa: E402


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
    args = ap.parse_args()

    root = fl.download_structure(force=args.force)
    print_tree(root)

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
