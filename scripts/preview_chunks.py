"""Phase 1C (CPU): browse filled tables for manual QA authoring.

Triggered from the notebook:
    !python scripts/preview_chunks.py --limit 15 --seed 7
    %run scripts/preview_chunks.py --limit 15 --seed 7 --format image --display

Prints or renders a seeded sample of GT-filled tables, with their sample_id, so you can
read off true answers and write the manual + unanswerable questions for
qa/qa_manual_seed.jsonl. GT-filled is used because gold answers come from GT (P4).

Use --format image --display through IPython %run for readable inline PNG tables. Running
through !python still writes the PNG files, but notebook inline display is not available
from that child process.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src import visualisation as vis  # noqa: E402
from src.table_chunk import chunk_id_for  # noqa: E402
from src.table_serialize import serialize_markdown  # noqa: E402

SOURCE_DIRS = {"gt": config.TABLES_GT_FILLED, "ocr": config.TABLES_OCR_FILLED}


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _preview_out_dir(source: str, seed: int, limit: int, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir)
    return config.FIGURES / "phase1c_preview" / f"{source}_seed{seed}_limit{limit}"


def _write_index(out_dir: Path, rows: list[dict]) -> Path:
    cards = []
    for row in rows:
        rel = row["table_path"].name
        title = html.escape(row["title"])
        cards.append(
            "<section>"
            f"<h2>{title}</h2>"
            f"<img src='{html.escape(rel)}' alt='{title}'>"
            "</section>"
        )
    body = "\n".join(cards)
    index = out_dir / "index.html"
    index.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}"
        "section{margin:0 0 32px 0;padding-bottom:28px;border-bottom:1px solid #d1d5db}"
        "h2{font-size:16px;margin:0 0 12px 0;font-weight:600}"
        "img{max-width:none;background:white;border:1px solid #d1d5db}"
        "</style>"
        f"{body}",
        encoding="utf-8",
    )
    return index


def _display_images(paths: list[Path]) -> None:
    try:
        from IPython.display import Image, display
    except ImportError:
        print("[display skipped] IPython is not available")
        return
    for path in paths:
        display(Image(filename=str(path)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["gt", "ocr"], default="gt")
    ap.add_argument("--limit", type=int, default=15, help="how many tables to show")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--format", choices=["text", "image", "both"], default="text",
                    help="text prints markdown; image writes PNG previews")
    ap.add_argument("--display", action="store_true",
                    help="display PNG previews inline; use IPython %%run, not !python")
    ap.add_argument("--out-dir", default=None,
                    help="where image previews are written")
    args = ap.parse_args()

    src_dir = SOURCE_DIRS[args.source]
    if not src_dir.exists():
        raise SystemExit(f"no {args.source}_filled tables at {src_dir}")

    paths = sorted(src_dir.glob("*.json"))
    random.Random(args.seed).shuffle(paths)
    paths = paths[:args.limit]

    rendered = []
    out_dir = _preview_out_dir(args.source, args.seed, args.limit, args.out_dir)
    if args.format in {"image", "both"}:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(paths, start=1):
        table = json.loads(path.read_text(encoding="utf-8"))
        sample_id = table.get("meta", {}).get("sample_id", path.stem)
        cid = chunk_id_for(sample_id)
        shape = f"{table.get('num_rows', 0)}x{table.get('num_cols', 0)}"

        if args.format in {"text", "both"}:
            print("=" * 100)
            print(f"sample_id : {sample_id}")
            print(f"chunk_id  : {cid}   ({shape})")
            print(serialize_markdown(table) or "(empty)")
            print()

        if args.format in {"image", "both"}:
            title = f"{i:02d}. {sample_id} | {cid} | {shape}"
            table_path = out_dir / f"{i:02d}_{_safe_name(sample_id)}_table.png"
            vis.render_table_image(table, title=title).save(table_path)
            rendered.append({"title": title, "table_path": table_path})

    if args.format in {"image", "both"}:
        index = _write_index(out_dir, rendered)
        print(f"wrote {len(rendered)} table images -> {out_dir}")
        print(f"index -> {index}")
        if args.display:
            _display_images([row["table_path"] for row in rendered])


if __name__ == "__main__":
    main()
