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

GPU steps (TATR, OCR, FAISS) run on a Colab VM, which installs:

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

Phase 0 (repo foundation) and Phase 1A-local (CPU table-topology logic) are done:
package skeleton, `config.py` with Colab/local path detection, the canonical table
schema, failure logging, financial number normalization, the TATR post-processing logic
(grid derivation, spanning-cell mapping, grid validation, occupancy-aware HTML parsing,
annotation gate), and the synthetic unit tests. Phase 1A-colab (real TATR inference +
topology metrics) is next. See [PLAN.md](PLAN.md) for the phase roadmap.
