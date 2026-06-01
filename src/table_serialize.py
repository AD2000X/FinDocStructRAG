"""Canonical table serialization for RAG chunks (Phase 1C).

Two serializations, compared head-to-head (PLAN Phase 1C, DESIGN_SPEC §7):

- **markdown** - a GitHub-flavored table grid. Structure-faithful and human-readable, but
  a flat value carries no explicit link to its row/column headers.
- **linearized** - one line per body row, each value paired with its column header
  ("Revenue: 2018 = 13,223; 2017 = 10,376"), so header->value associations survive in the
  retrieval/answer text. A spanning header applies to each column it covers.

Pure CPU, no model. Cells are placed by their (row_start, col_start) anchor; spanned
positions are filled by the covering cell (mirrors visualisation.topology_to_html).
"""

from __future__ import annotations

SERIALIZE_MARKDOWN = "markdown"
SERIALIZE_LINEARIZED = "linearized"
SERIALIZATIONS = (SERIALIZE_MARKDOWN, SERIALIZE_LINEARIZED)


def _covering_grid(table):
    """grid[r][c] = the cell covering (r, c) (anchor or spanned), or None."""
    n_rows = table.get("num_rows", 0)
    n_cols = table.get("num_cols", 0)
    grid = [[None] * n_cols for _ in range(n_rows)]
    for cell in table.get("cells", []):
        for r in range(cell["row_start"], min(cell["row_end"], n_rows)):
            for c in range(cell["col_start"], min(cell["col_end"], n_cols)):
                grid[r][c] = cell
    return grid, n_rows, n_cols


def _cell_text(cell) -> str:
    return (cell.get("text", "") or "").strip() if cell else ""


def _header_rows(grid, n_rows, n_cols) -> list[int]:
    """Rows that are column headers: any covering cell in the row is a header cell."""
    return [
        r for r in range(n_rows)
        if any(grid[r][c] is not None and grid[r][c].get("is_header")
               for c in range(n_cols))
    ]


def _column_headers(grid, header_rows, n_cols) -> list[str]:
    """Per-column header text, joining multi-row headers and de-duplicating spans."""
    headers = []
    for c in range(n_cols):
        parts = []
        for r in header_rows:
            t = _cell_text(grid[r][c])
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        headers.append(" ".join(parts))
    return headers


def table_grid(table):
    """Shared structural view of a table, reused by serialization and QA generation.

    Returns (grid, n_rows, n_cols, header_rows, col_headers): grid[r][c] is the cell
    covering (r, c) or None; header_rows are the column-header row indices; col_headers[c]
    is the joined header text for column c.
    """
    grid, n_rows, n_cols = _covering_grid(table)
    header_rows = _header_rows(grid, n_rows, n_cols)
    col_headers = _column_headers(grid, header_rows, n_cols)
    return grid, n_rows, n_cols, header_rows, col_headers


def serialize_markdown(table) -> str:
    """Render the table as a GitHub-flavored markdown grid.

    Row 0 is the markdown header row (GFM allows one); any further header rows render as
    body rows. Markdown has no colspan, so a spanning cell's text sits at its anchor column
    and the columns it covers are left blank.
    """
    grid, n_rows, n_cols = _covering_grid(table)
    if n_rows == 0 or n_cols == 0:
        return ""

    def row_cells(r):
        out = []
        for c in range(n_cols):
            cell = grid[r][c]
            anchored = (cell is not None
                        and cell["row_start"] == r and cell["col_start"] == c)
            out.append(_cell_text(cell).replace("|", r"\|") if anchored else "")
        return out

    lines = ["| " + " | ".join(row_cells(0)) + " |",
             "| " + " | ".join(["---"] * n_cols) + " |"]
    lines += ["| " + " | ".join(row_cells(r)) + " |" for r in range(1, n_rows)]
    return "\n".join(lines)


def serialize_linearized(table) -> str:
    """Render the table as one line per body row, pairing each value with its column header.

    Header rows are consumed to build the per-column headers and are not emitted as their
    own lines. Column 0 of a body row is its row label. Empty values are skipped, so the
    text stays compact for retrieval.
    """
    grid, n_rows, n_cols, header_rows, col_headers = table_grid(table)
    if n_rows == 0 or n_cols == 0:
        return ""
    header_set = set(header_rows)

    lines = []
    for r in range(n_rows):
        if r in header_set:
            continue
        label = _cell_text(grid[r][0])
        pairs = []
        for c in range(1, n_cols):
            val = _cell_text(grid[r][c])
            if not val:
                continue
            head = col_headers[c]
            pairs.append(f"{head} = {val}" if head else val)
        if not label and not pairs:
            continue
        prefix = f"{label}: " if label else ""
        lines.append(prefix + "; ".join(pairs))
    return "\n".join(lines)


def serialize(table, mode: str = SERIALIZE_LINEARIZED) -> str:
    """Serialize a canonical table in the named mode."""
    if mode == SERIALIZE_MARKDOWN:
        return serialize_markdown(table)
    if mode == SERIALIZE_LINEARIZED:
        return serialize_linearized(table)
    raise ValueError(f"unknown serialization mode: {mode!r}")
