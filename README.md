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

Phases 0 through 1C are complete; the v1 release (table-only RAG) is merged to `main`.
Delivered: the repo foundation; Phase 1A table topology (TATR grid derivation,
spanning-cell mapping, grid validation, occupancy-aware HTML parsing); Phase 1B OCR
content extraction (word-to-cell assignment, financial number normalization, content
metrics); and Phase 1C table-only RAG (BM25 + dense BGE cosine + RRF retrieval, one
chunk per table, single-provider grounded answer generation, GT-filled vs OCR-filled
corpora scored separately). Next is Phase 2 (DocLayNet layout integration: page-level
region detection -> table crop -> the existing Phase 1A/1B pipeline). See
[PLAN.md](PLAN.md) for the phase roadmap.
