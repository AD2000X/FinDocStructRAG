"""Phase 1A deliverable visualisations (DESIGN_SPEC 5.14).

Pure rendering helpers: draw TATR boxes / the derived cell grid over a crop, and render
the canonical table, metrics summary, and geometry flags as HTML/text. No torch, no
model - the Colab runner (scripts/render_phase1a_figures.py) loads the crop + artifacts
and calls these. The HTML/text/selection logic here is unit-tested; the PIL drawing is
exercised on Colab where the crops live.

P4: a TATR overlay is drawn from the raw TATR artifact (a prediction). It is never drawn
from the GT annotation; a GT overlay, if ever added, must be labelled as annotation.
"""

from __future__ import annotations

import html as _html

# Raw-artifact class key -> RGB, for the TATR box overlay (#2).
CLASS_COLORS = {
    "table_boxes": (127, 127, 127),
    "row_boxes": (31, 119, 180),
    "col_boxes": (255, 127, 14),
    "column_headers": (44, 160, 44),
    "projected_row_headers": (148, 103, 189),
    "spanning_cells": (214, 39, 40),
}
OVERLAY_KEYS = (
    "row_boxes", "col_boxes", "column_headers",
    "projected_row_headers", "spanning_cells",
)
# Human-readable class names for the overlay legend.
LEGEND_LABELS = {
    "row_boxes": "table row",
    "col_boxes": "table column",
    "column_headers": "column header",
    "projected_row_headers": "projected row header",
    "spanning_cells": "spanning cell",
}
CELL_COLOR = (31, 119, 180)
SPANNING_COLOR = (214, 39, 40)


def _draw_rect(draw, bbox, color, width=2, label=None):
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        draw.text((x1 + 2, y1 + 2), label, fill=color)


def _legend_strip(width, keys, swatch=14, pad=6, line_h=20):
    """A white strip captioning each drawn class with its colour swatch."""
    from PIL import Image, ImageDraw

    rows = [k for k in keys if k in LEGEND_LABELS]
    strip = Image.new("RGB", (width, pad * 2 + line_h * max(len(rows), 1)), "white")
    draw = ImageDraw.Draw(strip)
    y = pad
    for k in rows:
        color = CLASS_COLORS.get(k, (0, 0, 0))
        draw.rectangle([pad, y + 2, pad + swatch, y + 2 + swatch],
                       fill=color, outline=(0, 0, 0))
        draw.text((pad + swatch + pad, y + 3), LEGEND_LABELS[k], fill=(0, 0, 0))
        y += line_h
    return strip


def draw_tatr_overlay(image, raw_artifact, keys=OVERLAY_KEYS, width=2, legend=True):
    """#2: raw TATR boxes over a copy of the crop, colour-coded by class (+score).

    With legend=True, a caption strip listing the classes actually drawn is appended
    below the crop, so a viewer does not have to guess what each colour means.
    """
    from PIL import Image, ImageDraw

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    present = []
    for key in keys:
        boxes = raw_artifact.get(key, [])
        if boxes:
            present.append(key)
        color = CLASS_COLORS.get(key, (0, 0, 0))
        for box in boxes:
            score = box.get("score")
            label = f"{score:.2f}" if isinstance(score, (int, float)) else None
            _draw_rect(draw, box["bbox"], color, width, label)

    if not (legend and present):
        return img
    strip = _legend_strip(img.width, present)
    out = Image.new("RGB", (img.width, img.height + strip.height), "white")
    out.paste(img, (0, 0))
    out.paste(strip, (0, img.height))
    return out


def is_spanning(cell) -> bool:
    return (cell["row_end"] - cell["row_start"] > 1) or (
        cell["col_end"] - cell["col_start"] > 1
    )


def draw_cell_grid(image, table, width=2):
    """#3: the derived cell grid; spanning cells (span > 1) emphasised."""
    from PIL import ImageDraw

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for c in table.get("cells", []):
        if "bbox" not in c:
            continue
        spanning = is_spanning(c)
        color = SPANNING_COLOR if spanning else CELL_COLOR
        _draw_rect(draw, c["bbox"], color, width + 1 if spanning else width)
    return img


def draw_spanning_cells(image, table, width=3):
    """#4: only the spanning cells, labelled with their grid coordinates."""
    from PIL import ImageDraw

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for c in table.get("cells", []):
        if "bbox" not in c or not is_spanning(c):
            continue
        label = f"r{c['row_start']}-{c['row_end']} c{c['col_start']}-{c['col_end']}"
        _draw_rect(draw, c["bbox"], SPANNING_COLOR, width, label)
    return img


def topology_to_html(table) -> str:
    """#7: canonical table structure as an HTML grid.

    Cells are placed by their (row_start, col_start) anchor; row/col spans become
    rowspan/colspan. Text is usually empty for a topology-only prediction, so this shows
    structure, not content.
    """
    n_rows = table.get("num_rows", 0)
    n_cols = table.get("num_cols", 0)
    anchored = {(c["row_start"], c["col_start"]): c for c in table.get("cells", [])}
    occupied: set[tuple[int, int]] = set()

    rows_html = []
    for r in range(n_rows):
        tds = []
        col = 0
        while col < n_cols:
            if (r, col) in occupied:
                col += 1
                continue
            cell = anchored.get((r, col))
            if cell is None:
                tds.append("<td></td>")
                col += 1
                continue
            rs = cell["row_end"] - cell["row_start"]
            cs = cell["col_end"] - cell["col_start"]
            for rr in range(cell["row_start"], cell["row_end"]):
                for cc in range(cell["col_start"], cell["col_end"]):
                    occupied.add((rr, cc))
            tag = "th" if cell.get("is_header") else "td"
            attrs = (f' rowspan="{rs}"' if rs > 1 else "")
            attrs += (f' colspan="{cs}"' if cs > 1 else "")
            text = _html.escape(cell.get("text", "") or "")
            tds.append(f"<{tag}{attrs}>{text}</{tag}>")
            col += cs
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    return (
        "<table border='1' style='border-collapse:collapse'>"
        + "".join(rows_html)
        + "</table>"
    )


def summary_to_html(summary, subset_note: str = "") -> str:
    """#9: topology metrics summary as a small HTML table, with a subset disclaimer."""
    note = f"<p><em>{_html.escape(subset_note)}</em></p>" if subset_note else ""
    rows = "".join(
        f"<tr><th style='text-align:left'>{_html.escape(str(k))}</th>"
        f"<td>{_html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
    )
    return (
        "<h3>Phase 1A topology metrics</h3>"
        + note
        + "<table border='1' style='border-collapse:collapse'>"
        + rows
        + "</table>"
    )


def geometry_report(raw_artifact) -> str:
    """#5: a text report of the grid-geometry validation for one sample."""
    gv = raw_artifact.get("geometry_validation", {})
    lines = [
        f"sample_id: {raw_artifact.get('sample_id')}",
        f"valid: {gv.get('valid')}",
        f"rows: {len(raw_artifact.get('row_boxes', []))}  "
        f"cols: {len(raw_artifact.get('col_boxes', []))}  "
        f"spanning: {len(raw_artifact.get('spanning_cells', []))}",
        "flags:",
    ]
    flags = gv.get("flags", [])
    lines += [f"  - {f}" for f in flags] if flags else ["  (none)"]
    return "\n".join(lines)


def is_failure_candidate(raw_artifact) -> bool:
    """#8 selection: a sample worth showing as a failure case has geometry flags."""
    return bool(raw_artifact.get("geometry_validation", {}).get("flags"))
