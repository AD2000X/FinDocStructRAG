#!/usr/bin/env python3
"""Phase 4 demo: artifact-backed Gradio app for the FinDocStructRAG capstone.

Serves the already-produced evaluation artifacts (metrics, table outputs, layout crops, FUNSD
results) and does live BM25 retrieval + (optional) grounded answer generation over the existing
table chunks. Nothing runs a live PDF pipeline. The app degrades gracefully on two axes:

- Retrieval stack: BM25 is the always-on CPU default; dense + RRF light up only when the
  embedding stack (sentence-transformers) is importable.
- Answer generation: enabled only when OPENROUTER_API_KEY is set; otherwise the Table QA tab
  still shows retrieval and the answer box explains the key is missing.

gradio (and the dense/torch stack) are imported lazily inside the functions that need them, never
at module top, from src/, or from tests, so pytest and core stay gradio-free. See
docs/phase4_brief.md.

Usage:
    python scripts/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util
import json
import os
from collections import Counter

from src import config
from src import table_serialize
from src.retrieval import BM25Index, rrf_fuse
from src.llm_client import build_openrouter_complete, generate_answer

CORPORA = ["gt_linearized", "gt_markdown", "ocr_linearized", "ocr_markdown"]
TOP_K = 5

HAS_KEY = bool(os.getenv("OPENROUTER_API_KEY"))
DENSE_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
RETRIEVAL_METHODS = ["bm25"] + (["dense", "rrf"] if DENSE_AVAILABLE else [])

_CHUNKS: dict = {}
_BM25: dict = {}
_DENSE: dict = {}
_EMBEDDER = None


# --- artifact loading (cached) ---


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_chunks(corpus: str) -> list[dict]:
    if corpus not in _CHUNKS:
        path = config.CHUNKS / f"{corpus}.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        _CHUNKS[corpus] = rows
    return _CHUNKS[corpus]


def _chunk_by_id(corpus: str) -> dict:
    return {c["chunk_id"]: c for c in load_chunks(corpus)}


def get_bm25(corpus: str):
    if corpus not in _BM25:
        chunks = load_chunks(corpus)
        _BM25[corpus] = BM25Index(chunks) if chunks else None
    return _BM25[corpus]


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from src.dense_retrieval import build_bge_embedder
        _EMBEDDER = build_bge_embedder()
    return _EMBEDDER


def get_dense(corpus: str):
    if corpus not in _DENSE:
        from src.dense_retrieval import DenseIndex
        chunks = load_chunks(corpus)
        _DENSE[corpus] = DenseIndex(chunks, _get_embedder()) if chunks else None
    return _DENSE[corpus]


# --- Table QA ---


def retrieve_ids(corpus: str, query: str, method: str, top_k: int = TOP_K):
    """Return (chunk_ids, note). Falls back to BM25 if dense/RRF is unavailable or errors."""
    bm = get_bm25(corpus)
    if bm is None:
        return [], f"No chunks for corpus '{corpus}'."
    if method == "bm25":
        return bm.search(query, top_k), ""
    try:
        dn = get_dense(corpus)
        if method == "dense":
            return dn.search(query, top_k), ""
        fused = rrf_fuse([bm.search(query, top_k * 2), dn.search(query, top_k * 2)], top_k=top_k)
        return fused, ""
    except Exception as e:
        return bm.search(query, top_k), f"(dense/RRF unavailable: {type(e).__name__}; showing BM25)"


def _render_chunks(corpus: str, chunk_ids: list[str]) -> str:
    by_id = _chunk_by_id(corpus)
    out = []
    for rank, cid in enumerate(chunk_ids, 1):
        c = by_id.get(cid, {})
        out.append(f"**{rank}. `{cid}`**  (source={c.get('text_source', '?')}, "
                   f"{c.get('serialization', '?')})\n\n```\n{c.get('text', '')}\n```")
    return "\n\n".join(out) if out else "_No results._"


def qa_retrieve(corpus: str, query: str, method: str) -> str:
    if not (query or "").strip():
        return "_Enter a question._"
    ids, note = retrieve_ids(corpus, query, method)
    return f"**Corpus:** `{corpus}`  **Method:** {method}  {note}\n\n" + _render_chunks(corpus, ids)


def qa_answer(corpus: str, query: str, method: str) -> str:
    if not HAS_KEY:
        return ("_Answer generation disabled: set `OPENROUTER_API_KEY` to enable. "
                "Retrieval above still works._")
    if not (query or "").strip():
        return "_Enter a question._"
    ids, _ = retrieve_ids(corpus, query, method)
    by_id = _chunk_by_id(corpus)
    evidence = [by_id[c] for c in ids if c in by_id]
    try:
        ans = generate_answer(query, evidence, complete=build_openrouter_complete())
    except Exception as e:
        return f"_Answer generation error: {type(e).__name__}: {e}_"
    cites = ", ".join(f"`{c}`" for c in ans.citations) or "(none)"
    status = "abstained" if ans.abstained else "answered"
    return f"**Answer ({status}):** {ans.answer or '(empty)'}\n\n**Citations:** {cites}"


# --- Table Extraction ---


_TABLE_SOURCES = [
    (config.TABLES_GT_FILLED, "GT-filled (reference, P4)"),
    (config.TABLES_OCR_FILLED, "OCR-filled (real extraction)"),
    (config.TABLES_TATR_PREDICTED, "TATR-predicted (structure)"),
]


def list_samples() -> list[str]:
    d = config.TABLES_GT_FILLED
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def table_view(sample_id: str) -> str:
    if not sample_id:
        return "_Pick a sample._"
    parts = []
    for d, label in _TABLE_SOURCES:
        tbl = _load_json(d / f"{sample_id}.json")
        if tbl is None:
            parts.append(f"### {label}\n_Not available._")
        else:
            parts.append(f"### {label}  ({tbl.get('num_rows', '?')}x{tbl.get('num_cols', '?')})\n\n"
                         + (table_serialize.serialize_markdown(tbl) or "_empty_"))
    return "\n\n".join(parts)


# --- Layout ---


def list_layout_pages() -> list[str]:
    d = config.LAYOUT_OUTPUT / "regions"
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


def layout_summary_md() -> str:
    s = _load_json(config.EVALUATION / "phase4_summary.json")
    g = (s or {}).get("layout", {})
    if not g.get("available"):
        return "_Phase 2 layout summary not available (run build_phase4_summary)._"
    cr = g["crop_to_tatr"]
    return (f"**Phase 2 layout (aggregate):** mean crop IoU {g['mean_crop_iou']:.3f}; "
            f"matched@0.50 recall {g['matched@0.50']['recall']:.3f}; "
            f"table-free FP rate {g['crop_fp_rate']:.3f}; "
            f"crop->TATR OK {cr['ok']}/{cr['n']}.")


def layout_view(page_id: str):
    if not page_id:
        return [], "_Pick a page._"
    crops = sorted(str(p) for p in (config.LAYOUT_OUTPUT / "crops").glob(f"{page_id}_table_*.png"))
    regions = _load_json(config.LAYOUT_OUTPUT / "regions" / f"{page_id}.json") or []
    counts = Counter(r["label"] for r in regions)
    tables = [r for r in regions if r["label"] == "table"]
    lines = [f"**Page `{page_id}`** - {len(regions)} regions, {len(crops)} table crop(s).",
             "region counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))]
    for i, t in enumerate(tables):
        lines.append(f"- table {i}: score {t.get('score', '?')}, box {[round(x, 1) for x in t['box']]}")
    return crops, "\n".join(lines)


# --- FUNSD + Overview + Limitations ---


def gradio_allowed_paths() -> list[str]:
    """Artifact paths Gradio may serve when outputs live outside the repo, e.g. Colab Drive."""
    paths = [
        config.LAYOUT_OUTPUT / "crops",
    ]
    return [str(p.resolve()) for p in paths if p.exists()]


def funsd_view() -> str:
    d = _load_json(config.EVALUATION / "phase3_funsd_relations.json")
    if d is None:
        return "_FUNSD results not available._"
    lines = [f"**Held-out headline:** `{d['primary']}`.", "",
             "| split | scope | precision | recall | f1 |", "|---|---|---|---|---|"]
    for split, scopes in d["results"].items():
        for scope, m in scopes.items():
            lines.append(f"| {split} | {scope} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines += ["", "_Annotation-only deterministic baseline; recall is the design ceiling. "
              "Qualitative error overlays live in `notebooks/05_phase3_funsd_relations.ipynb`._"]
    return "\n".join(lines)


def overview_view() -> str:
    summary = _load_json(config.EVALUATION / "phase4_summary.json")
    parts = ["## Capstone overview", ""]
    if summary:
        parts.append("**Artifact availability:** " + ", ".join(
            f"{name}={'OK' if part.get('available') else 'MISSING'}" for name, part in summary.items()))
        f = summary.get("funsd", {})
        if f.get("available"):
            h = f["headline"]
            parts.append(f"\n**FUNSD headline ({f['primary']}):** "
                         f"P {h['precision']:.3f} / R {h['recall']:.3f} / F1 {h['f1']:.3f}")
    else:
        parts.append("_Run `python scripts/build_phase4_summary.py` to generate the summary._")
    metrics_md = config.ROOT / "reports" / "phase4_metrics.md"
    if metrics_md.exists():
        parts += ["", "---", "", metrics_md.read_text(encoding="utf-8")]
    return "\n".join(parts)


LIMITATIONS_MD = """## Limitations (honest scope)

- **Subset evaluation**, not whole-dataset benchmarks.
- **GT-filled vs OCR-filled are kept separate (P4):** GT-filled is a QA-validation reference,
  never reported as an extraction output; OCR-filled is the real extraction.
- **FUNSD V1 is annotation-only** over GT entities (no entity detection, single link per answer,
  geometry-only) - recall is the design ceiling.
- **RAG is table-only**; full-document text, charts, and figures are out of scope
  (caption-level / future work).
- **This demo is artifact-backed** - it serves produced outputs and does live retrieval / answer
  generation over the existing chunks; it does not run a live PDF -> pipeline.
- **GriTS / Ragas / DeepEval** are future work, not used as gates here.
"""


def main() -> None:
    import gradio as gr

    samples = list_samples()
    pages = list_layout_pages()
    answer_gen = "enabled" if HAS_KEY else "disabled (no OPENROUTER_API_KEY)"

    with gr.Blocks(title="FinDocStructRAG capstone demo") as demo:
        gr.Markdown(f"# FinDocStructRAG - capstone demo\n"
                    f"Artifact-backed. Retrieval: {', '.join(RETRIEVAL_METHODS)}. "
                    f"Answer generation: {answer_gen}.")

        with gr.Tab("Overview"):
            ov = gr.Markdown()
            demo.load(overview_view, None, ov)

        with gr.Tab("Table QA"):
            with gr.Row():
                corpus = gr.Dropdown(CORPORA, value="gt_linearized", label="corpus (GT vs OCR)")
                method = gr.Dropdown(RETRIEVAL_METHODS, value="bm25", label="retrieval method")
            question = gr.Textbox(label="question",
                                  placeholder="e.g. What was the discount rate in 2014?")
            with gr.Row():
                btn_r = gr.Button("Retrieve")
                btn_a = gr.Button("Generate answer" + ("" if HAS_KEY else " (disabled)"))
            results = gr.Markdown()
            answer = gr.Markdown()
            btn_r.click(qa_retrieve, [corpus, question, method], results)
            btn_a.click(qa_answer, [corpus, question, method], answer)

        with gr.Tab("Table Extraction"):
            if samples:
                sample = gr.Dropdown(samples, value=samples[0], label="sample table")
                tbl_out = gr.Markdown()
                sample.change(table_view, sample, tbl_out)
                demo.load(table_view, sample, tbl_out)
            else:
                gr.Markdown("_No table outputs found under outputs/tables/._")

        with gr.Tab("Layout"):
            gr.Markdown(layout_summary_md())
            if pages:
                page = gr.Dropdown(pages, value=pages[0], label="DocLayNet page")
                gallery = gr.Gallery(label="table crops")
                region_md = gr.Markdown()
                page.change(layout_view, page, [gallery, region_md])
                demo.load(layout_view, page, [gallery, region_md])
            else:
                gr.Markdown("_No layout crops found under outputs/layout/._")

        with gr.Tab("FUNSD Relations"):
            funsd_md = gr.Markdown()
            demo.load(funsd_view, None, funsd_md)

        with gr.Tab("Limitations"):
            gr.Markdown(LIMITATIONS_MD)

    demo.launch(share=config.IN_COLAB, allowed_paths=gradio_allowed_paths())


if __name__ == "__main__":
    main()
