"""FinTabNet.c loading and structure-annotation parsing.

The dataset (bsmock/FinTabNet.c on HuggingFace) is not a load_dataset() dataset: it
ships as tar.gz archives. We download the structure archive via huggingface_hub and
parse its PASCAL VOC XML annotations (the standard Table Transformer format).

This runs on the Colab VM (network + Drive cache). The parser itself is pure and could
be unit-tested with a synthetic XML string, but the exact on-disk layout and class
strings are confirmed by scripts/inspect_fintabnet.py on the first real run.
"""

from __future__ import annotations

import json
import random
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

from . import config

REPO_ID = "bsmock/FinTabNet.c"
STRUCTURE_ARCHIVE = "FinTabNet.c-Structure.tar.gz"
# Per-sample GT word tokens live alongside images, confirmed by inspect_fintabnet
# --gt-text: FinTabNet.c-Structure/words/<stem>_words.json.
STRUCTURE_SUBDIR = "FinTabNet.c-Structure"
WORDS_SUFFIX = "_words.json"

# Standard Table Transformer structure classes (PASCAL VOC <name> values).
# Verified against the real archive on first run; adjust here if they differ.
CLS_TABLE = "table"
CLS_COLUMN = "table column"
CLS_ROW = "table row"
CLS_COLUMN_HEADER = "table column header"
CLS_PROJECTED_ROW_HEADER = "table projected row header"
CLS_SPANNING_CELL = "table spanning cell"


def _dataset_cache_root() -> Path:
    """Where to extract the raw dataset.

    On Colab this is scratch (/content), not Drive: the archive is cheaply
    re-downloadable from HF, and extracting thousands of small files to Drive is very
    slow. Only expensive pipeline outputs persist to Drive (config.OUTPUT_ROOT).
    """
    if config.IN_COLAB:
        return Path("/content") / "fintabnet_c_cache"
    return config.DATA_ROOT / "raw" / "fintabnet_c"


def structure_root() -> Path:
    """Directory where the structure archive is extracted."""
    return _dataset_cache_root() / "structure"


def download_structure(force: bool = False) -> Path:
    """Download and extract FinTabNet.c-Structure.tar.gz. Idempotent.

    Returns the extraction directory. Skips work if already extracted unless force.
    """
    from huggingface_hub import hf_hub_download

    dest = structure_root()
    marker = dest / ".extracted"
    if marker.exists() and not force:
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(
        repo_id=REPO_ID,
        filename=STRUCTURE_ARCHIVE,
        repo_type="dataset",
    )
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    marker.write_text("ok", encoding="utf-8")
    return dest


def find_xml_files(
    root: Path, limit: int | None = None, seed: int | None = None
) -> list[Path]:
    """PASCAL VOC annotation files under root (recursive).

    Default order is sorted-by-path (deterministic, but alphabetically-first files are
    issuer-biased). Pass a seed to draw a reproducible random sample instead: the full
    list is shuffled with that seed before slicing, so seed=42 with limit 10/50/300
    yields nested subsets (10 ⊂ 50 ⊂ 300) - a smaller run's samples are reused when the
    limit grows, keeping the resumable manifest valid.
    """
    files = sorted(root.rglob("*.xml"))
    if seed is not None and limit:
        random.Random(seed).shuffle(files)
    return files[:limit] if limit else files


def _bbox(obj: ET.Element) -> list[float]:
    b = obj.find("bndbox")
    return [
        float(b.findtext("xmin")),
        float(b.findtext("ymin")),
        float(b.findtext("xmax")),
        float(b.findtext("ymax")),
    ]


def parse_structure_xml(xml_path: str | Path) -> dict:
    """Parse one PASCAL VOC structure annotation into grouped boxes.

    Returns a dict shaped for normalize_tatr_prediction()/boxes_to_grid():
        row_boxes, col_boxes, spanning_cells -> [{"bbox": [...]}, ...]
        column_headers, projected_row_headers -> [{"bbox": [...]}, ...]
        table_bbox -> [...] or None
        image_filename -> str
        class_counts -> {class_name: count}  (for format verification)
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    grouped: dict[str, list[dict]] = {}
    class_counts: dict[str, int] = {}
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        class_counts[name] = class_counts.get(name, 0) + 1
        grouped.setdefault(name, []).append({"bbox": _bbox(obj)})

    table_boxes = grouped.get(CLS_TABLE, [])
    return {
        "image_filename": root.findtext("filename"),
        "table_bbox": table_boxes[0]["bbox"] if table_boxes else None,
        "row_boxes": grouped.get(CLS_ROW, []),
        "col_boxes": grouped.get(CLS_COLUMN, []),
        "spanning_cells": grouped.get(CLS_SPANNING_CELL, []),
        "column_headers": grouped.get(CLS_COLUMN_HEADER, []),
        "projected_row_headers": grouped.get(CLS_PROJECTED_ROW_HEADER, []),
        "class_counts": class_counts,
    }


def words_path_for(sample_id: str) -> Path:
    """Path to a sample's GT word-token JSON (FinTabNet.c-Structure/words/...).

    sample_id is the shared stem of the xml/jpg (e.g. "AAL_2002_page_41_table_1").
    """
    return structure_root() / STRUCTURE_SUBDIR / "words" / f"{sample_id}{WORDS_SUFFIX}"


def parse_words_json(source: str | Path | list) -> list[dict]:
    """FinTabNet.c per-sample words JSON -> word dicts for assign_words_to_cells().

    Each GT record carries a bbox [x1,y1,x2,y2] in the crop's pixel coordinates (the
    same space as the structure XML) and text. We keep only what assignment needs and
    tag the source as GT (confidence 1.0). Accepts a path or an already-loaded list.
    """
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        data = source
    return [
        {
            "text": w.get("text", ""),
            "bbox": [float(v) for v in w["bbox"]],
            "confidence": 1.0,
            "source": "gt",
        }
        for w in data
    ]
