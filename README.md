# FinDocStructRAG

Layout-aware financial document intelligence pipeline: PDF layout parsing, financial
table extraction, OCR/form relation linking, and retrieval-grounded QA under Colab T4
constraints.

- **What to build** (schemas, function signatures, metrics, tests): [DESIGN_SPEC.md](DESIGN_SPEC.md)
- **Order, environment workflow, acceptance criteria**: [PLAN.md](PLAN.md)
- **Contributor rules**: [CLAUDE.md](CLAUDE.md)

## Install

Local development (CPU logic + tests, no GPU):

```
pip install -r requirements.txt
```

GPU steps (TATR, OCR, dense embedding) run on a Colab VM, which installs:

```
pip install -r requirements-colab.txt
```

## Repository layout

```
src/        core logic (functions, schemas, parsers, metrics) — the source of truth
scripts/    repeatable pipeline runners (CLI)
tests/      pytest unit / smoke tests
notebooks/  Colab GPU runners + demo/report only (no logic)
data/        datasets (gitignored; kept on Google Drive in Colab)
outputs/     pipeline outputs (gitignored; kept on Google Drive in Colab)
assets/      static assets
reports/     final report artifacts
```

## Development loop

Core logic is `.py` under `src/`; notebooks only mount Drive, `git pull`, install, and
run. Edit locally in VS Code, push to GitHub, and Colab pulls the same code onto the VM
(the Colab kernel cannot see local files). Run the local CPU tests with:

```
pytest
```

## Status

**Phases 0 through 4 are complete and merged to `main`.** Delivered: the repo foundation;
Phase 1A table topology (TATR grid derivation, spanning-cell mapping, grid validation,
occupancy-aware HTML parsing); Phase 1B OCR content extraction (word-to-cell assignment,
financial number normalization, content metrics); Phase 1C table-only RAG (BM25 + dense
BGE cosine + RRF retrieval, one chunk per table, single-provider grounded answer
generation, GT-filled vs OCR-filled corpora scored separately); Phase 2 DocLayNet
layout-crop integration (page-level region detection -> table crop -> the Phase 1A/1B
pipeline); and Phase 3 FUNSD relation-linking baseline (annotation-only deterministic
predictor, held-out `test_50.qa_links` F1 0.727); and Phase 4 final integration — one
generated evaluation summary, an artifact-backed key-optional Gradio demo, and a written
report (no new research; GriTS / Ragas / DeepEval are future work).

Entry points:
- `python scripts/build_phase4_summary.py` -> `outputs/evaluation/phase4_summary.json` +
  the committed `reports/phase4_metrics.md` (generated metrics; never hand-copied).
- `reports/final_report.md` / `notebooks/07_final_report.ipynb` — the final report.
- `python scripts/run_demo.py` / `notebooks/06_demo.ipynb` — the Gradio demo (launches with
  no API key: BM25 retrieval + metrics + artifact views; answer generation needs
  `OPENROUTER_API_KEY`).

See [PLAN.md](PLAN.md) for the phase roadmap.
