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
import math
import textwrap

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


def draw_ocr_debug_overlay(image, gt_table, ocr_table, width=2):
    """Phase 1B content debug: GT cell boxes (green), TATR pred cell boxes (gray) and OCR
    word/detection boxes (red) on one crop, to see whether OCR boxes span GT column
    boundaries (a column-grouping error) or the pred columns are shifted.

    OCR word boxes are read from ocr_table cells' "words" (persisted in ocr_filled).
    """
    from PIL import Image, ImageDraw

    gt_color = CLASS_COLORS["column_headers"]   # green
    pred_color = (127, 127, 127)                # gray
    word_color = SPANNING_COLOR                 # red

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for c in gt_table.get("cells", []):
        if "bbox" in c:
            _draw_rect(draw, c["bbox"], gt_color, width)
    for c in ocr_table.get("cells", []):
        if "bbox" in c:
            _draw_rect(draw, c["bbox"], pred_color, 1)
        for w in c.get("words", []):
            if "bbox" in w:
                _draw_rect(draw, w["bbox"], word_color, width)

    legend = [(gt_color, "GT cell"), (pred_color, "TATR pred cell"),
              (word_color, "OCR word")]
    strip_h = 6 + 20 * len(legend) + 6
    strip = Image.new("RGB", (img.width, strip_h), "white")
    d2 = ImageDraw.Draw(strip)
    for i, (color, label) in enumerate(legend):
        y = 6 + i * 20
        d2.rectangle([6, y + 2, 20, y + 16], fill=color, outline=(0, 0, 0))
        d2.text((26, y + 3), label, fill=(0, 0, 0))
    out = Image.new("RGB", (img.width, img.height + strip_h), "white")
    out.paste(img, (0, 0))
    out.paste(strip, (0, img.height))
    return out


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


def _load_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"]
        if bold else ["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"]
    )
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _measure(draw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text or " ", font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _clean_cell_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _wrap_cell_text(draw, text: str, font, max_width: int) -> list[str]:
    """Wrap text to fit a rendered table cell, splitting long tokens only if needed."""
    text = _clean_cell_text(text)
    if not text:
        return [""]

    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        wrapped = textwrap.wrap(
            paragraph,
            width=max(8, len(paragraph)),
            break_long_words=False,
            break_on_hyphens=False,
        )
        words = " ".join(wrapped).split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if _measure(draw, candidate, font)[0] <= max_width:
                line = candidate
                continue
            if line:
                lines.append(line)
            if _measure(draw, word, font)[0] <= max_width:
                line = word
                continue
            chunk = ""
            for ch in word:
                candidate = chunk + ch
                if not chunk or _measure(draw, candidate, font)[0] <= max_width:
                    chunk = candidate
                else:
                    lines.append(chunk)
                    chunk = ch
            line = chunk
        if line:
            lines.append(line)
    return lines or [""]


def render_table_image(
    table,
    *,
    title: str = "",
    font_size: int = 14,
    min_col_width: int = 90,
    max_col_width: int = 260,
    padding: int = 8,
):
    """Render a canonical table as a readable PNG image.

    This is for notebook/manual-QA review. Unlike markdown, it preserves row/col spans
    visually and wraps long labels so wide financial tables remain readable.
    """
    from PIL import Image, ImageDraw

    n_rows = table.get("num_rows", 0)
    n_cols = table.get("num_cols", 0)
    if n_rows <= 0 or n_cols <= 0:
        return Image.new("RGB", (600, 80), "white")

    font = _load_font(font_size)
    header_font = _load_font(font_size, bold=True)
    title_font = _load_font(font_size + 2, bold=True)
    scratch = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(scratch)

    cells = table.get("cells", [])
    anchored = {(c["row_start"], c["col_start"]): c for c in cells}
    covered: set[tuple[int, int]] = set()
    for cell in cells:
        for r in range(cell["row_start"], min(cell["row_end"], n_rows)):
            for c in range(cell["col_start"], min(cell["col_end"], n_cols)):
                covered.add((r, c))

    col_widths = [min_col_width] * n_cols
    for cell in cells:
        cs = max(1, min(cell["col_end"], n_cols) - cell["col_start"])
        cell_font = header_font if cell.get("is_header") else font
        text = _clean_cell_text(cell.get("text", ""))
        if not text:
            continue
        longest_word = max(text.split(), key=len, default="")
        text_w = min(_measure(draw, text, cell_font)[0], max_col_width * cs)
        word_w = _measure(draw, longest_word, cell_font)[0]
        needed = min(max_col_width * cs, max(min_col_width * cs, text_w, word_w))
        per_col = math.ceil((needed + 2 * padding) / cs)
        for c in range(cell["col_start"], min(cell["col_end"], n_cols)):
            col_widths[c] = min(max_col_width, max(col_widths[c], per_col))

    line_h = max(_measure(draw, "Ag", font)[1], _measure(draw, "Ag", header_font)[1]) + 5
    min_row_height = line_h + 2 * padding
    row_heights = [min_row_height] * n_rows
    rendered_lines = {}
    required_heights = {}
    for cell in cells:
        rs = max(1, min(cell["row_end"], n_rows) - cell["row_start"])
        cell_font = header_font if cell.get("is_header") else font
        width = max(24, sum(col_widths[cell["col_start"]:min(cell["col_end"], n_cols)])
                    - 2 * padding)
        lines = _wrap_cell_text(draw, cell.get("text", ""), cell_font, width)
        required = max(min_row_height, len(lines) * line_h + 2 * padding)
        key = (cell["row_start"], cell["col_start"])
        rendered_lines[key] = lines
        required_heights[key] = required
        if rs == 1:
            row_heights[cell["row_start"]] = max(row_heights[cell["row_start"]], required)

    for cell in cells:
        rs = max(1, min(cell["row_end"], n_rows) - cell["row_start"])
        if rs == 1:
            continue
        key = (cell["row_start"], cell["col_start"])
        current = sum(row_heights[cell["row_start"]:min(cell["row_end"], n_rows)])
        deficit = required_heights[key] - current
        if deficit > 0:
            extra = math.ceil(deficit / rs)
            for r in range(cell["row_start"], min(cell["row_end"], n_rows)):
                row_heights[r] += extra

    margin = 18
    title_h = 0
    if title:
        title_h = _measure(draw, title, title_font)[1] + 14
    x_edges = [margin]
    for width in col_widths:
        x_edges.append(x_edges[-1] + width)
    y_edges = [margin + title_h]
    for height in row_heights:
        y_edges.append(y_edges[-1] + height)

    img = Image.new(
        "RGB",
        (x_edges[-1] + margin, y_edges[-1] + margin),
        "white",
    )
    draw = ImageDraw.Draw(img)
    if title:
        draw.text((margin, margin), title, fill=(20, 20, 20), font=title_font)

    border = (120, 120, 120)
    header_bg = (232, 238, 246)
    body_bg = (255, 255, 255)
    missing_bg = (248, 248, 248)

    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in covered:
                continue
            draw.rectangle(
                [x_edges[c], y_edges[r], x_edges[c + 1], y_edges[r + 1]],
                fill=missing_bg,
                outline=border,
            )

    for (r, c), cell in sorted(anchored.items()):
        if r >= n_rows or c >= n_cols:
            continue
        x1, x2 = x_edges[c], x_edges[min(cell["col_end"], n_cols)]
        y1, y2 = y_edges[r], y_edges[min(cell["row_end"], n_rows)]
        is_header = bool(cell.get("is_header"))
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=header_bg if is_header else body_bg,
            outline=border,
        )
        cell_font = header_font if is_header else font
        y = y1 + padding
        for line in rendered_lines.get((r, c), [""]):
            draw.text((x1 + padding, y), line, fill=(20, 20, 20), font=cell_font)
            y += line_h

    return img


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
