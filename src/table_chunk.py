"""Table chunk schema + builder for table-only RAG (Phase 1C).

One chunk per reconstructed table: the serialized table text plus provenance. Chunks are
the only retrieval unit in Phase 1C (table-only RAG; full-document corpus waits until after
Phase 2 - PLAN Phase 1C non-goal).

GT-filled and OCR-filled chunks are built into separate corpora (P4) and never mixed. The
chunk_id is corpus-independent (`table:<sample_id>`), so a QA relevance judgment holds
across the GT and OCR runs and across both serializations - only the corpus the chunk lives
in changes, not its identity.
"""

from __future__ import annotations

from typing import TypedDict

from .table_serialize import SERIALIZE_LINEARIZED, serialize

CHUNK_ID_PREFIX = "table:"


def chunk_id_for(sample_id: str) -> str:
    """Stable, corpus-independent chunk id for a table (used as the relevance judgment)."""
    return f"{CHUNK_ID_PREFIX}{sample_id}"


class TableChunk(TypedDict, total=False):
    chunk_id: str
    sample_id: str
    text_source: str       # gt | ocr (from the table's meta)
    serialization: str     # markdown | linearized
    text: str
    num_rows: int
    num_cols: int


def build_chunk(table, *, serialization: str = SERIALIZE_LINEARIZED) -> TableChunk:
    """Build one retrieval chunk from a filled canonical table.

    sample_id / text_source come from the table's meta (set by fill_table). text is the
    table serialized in the requested mode.
    """
    meta = table.get("meta", {})
    sample_id = meta.get("sample_id", "unknown")
    return {
        "chunk_id": chunk_id_for(sample_id),
        "sample_id": sample_id,
        "text_source": meta.get("text_source", "none"),
        "serialization": serialization,
        "text": serialize(table, serialization),
        "num_rows": table.get("num_rows", 0),
        "num_cols": table.get("num_cols", 0),
    }
