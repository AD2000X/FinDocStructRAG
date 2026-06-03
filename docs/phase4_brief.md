# Phase 4 — Final Demo + Eval Summary + Final Report

> Implementation brief for Phase 4. Committed in the repo (travels with `git pull` to Colab) so
> the references to it in `DEVLOG.md` and the `src/phase4_summary.py` /
> `scripts/build_phase4_summary.py` docstrings resolve. Status: PR-A/PR-B/PR-C are implemented
> on `feature/phase4-demo` — summary backbone, generated metrics, final report, report notebook,
> key-optional Gradio demo, and demo notebook.

## Context

Phases 0-3 are merged to `main` (FinTabNet.c table topology + OCR content + table-only RAG +
DocLayNet layout + FUNSD relations). Phase 4 is the **final integration**: make the work
presentable, reportable, and reproducible. It is explicitly **not new research** — it assembles the existing
deterministic/custom metrics into one summary, a Gradio demo, and a written report.
GriTS/Ragas/DeepEval are future work.

All Drive evaluation artifacts are staged locally under `outputs/` (gitignored): metrics JSONs,
layout CSVs, the RAG chunk corpus, QA sets, table outputs, crops/regions. FUNSD raw is at
`data/raw/funsd/`.

## Locked decisions

- **Assemble, don't research.** GriTS / Ragas / DeepEval = future work, never a Phase 4 gate.
- **Report is the product; notebooks are runners** (P1/P2): aggregation in `.py`; notebooks only
  pull branch + run a script + display tables/figures.
- **Demo is artifact-backed**, not live PDF -> layout -> TATR -> OCR -> RAG. The only live piece
  is a QA box doing retrieval + answer generation over the **existing** chunk corpus.
- **Notebook numbering 06/07** (contiguous). **Entrypoint `scripts/run_demo.py`** (runners live
  in `scripts/`; no root `app.py` unless HF Spaces later).
- **Demo degrades gracefully on two independent axes.** (a) *Retrieval stack:* default to
  **BM25 retrieval-only** (pure CPU, no model); enable dense + RRF only when the embedding stack
  is importable (a key-less reviewer may also lack a GPU). (b) *Answer generation:* gated solely
  by `OPENROUTER_API_KEY` (disabled tab + key-missing message when absent). The demo must fully
  launch with **neither**.
- **Report metrics generated from the summary, never hand-copied.** The builder emits
  `phase4_summary.json` and a paste-ready markdown table; report numbers read from the table.
- **Commit policy:** `reports/phase4_metrics.md` is committed (generated report snippet);
  `outputs/evaluation/phase4_summary.json` stays gitignored under `outputs/`. The no-drift gate
  checks `reports/phase4_metrics.md` is byte-identical after a rebuild (the builder writes LF).
- **Retrieval reported as hit@1 / hit@5 / hit@10 + MRR@10 only.** With one relevant chunk per
  question `recall@k == hit@k` (`src/eval_retrieval.py`), so recall@k is dropped from the report.

## Input artifacts (all verified present)

| Source | File | Headline keys |
|---|---|---|
| 1A topology | `outputs/evaluation/phase1a_topology_<run-id>.json` | row/col_count_accuracy, cell_occupancy_f1, spanning_cell_detection_rate (n=300) |
| 1B content | `outputs/evaluation/phase1b_content_<run-id>.json` | `aggregate` / `one_to_one` / `topology_matched_subset` cell metrics |
| 1C retrieval | `outputs/evaluation/rag/phase1c_retrieval.json` | corpora x {bm25,dense,rrf} x hit@{1,5,10}, mrr@10 |
| 1C QA | `outputs/evaluation/rag/phase1c_qa.json` | configs x {answer_exact, numeric_relaxed, citation_hit, abstain_rate} — GT vs OCR |
| 2 layout | `outputs/layout/diagnostic_pos.csv` + `diagnostic_neg.csv` + `smoke_structure.csv` | mean crop IoU, matched@0.50/0.75, table-free FP rate, crop->TATR OK rate |
| 3 FUNSD | `outputs/evaluation/phase3_funsd_relations.json` | `primary`="test_50.qa_links"; results[split][scope] P/R/F1 |

Default deliverable run-id is `mvp_rand` (Phase 1A/1B). **Phase 2 has no JSON**; the builder
aggregates it inline from the staged CSVs (no Colab re-run), matching the table-level matching +
FP definitions printed by `scripts/eval_layout_iou.py` (and `scripts/smoke_structure.py` for the
crop->TATR OK/WARN split). The inline aggregation reproduces the DEVLOG layout numbers exactly
(mean crop IoU 0.900; matched@0.50 0.900/0.916; matched@0.75 0.880/0.895; crop->TATR 285/286).

## Files

- `src/phase4_summary.py` (new) — pure helpers, no file/Drive/gradio IO: `summarize_topology` /
  `_content` / `_retrieval` (drops recall@k) / `_qa` / `_funsd` (headline from the JSON's own
  `primary` pointer); `layout_metrics_from_rows(pos, neg, smoke)` (aggregation over parsed CSV
  rows); `build_summary(parts)` (missing part -> `{"available": false}`);
  `render_metrics_markdown(summary)` (deterministic paste-ready table). Style of
  `src/eval_funsd.py`.
- `scripts/build_phase4_summary.py` (new) — reads the five JSONs + three layout CSVs, calls the
  pure helpers, writes `outputs/evaluation/phase4_summary.json` (gitignored) +
  `reports/phase4_metrics.md` (committed, LF). Graceful on a missing artifact.
- `scripts/run_demo.py` (new, PR-C) — Gradio app; `gradio` imported inside the script only (never
  from `src/` or tests); BM25 retrieval default, dense/RRF only if the embedding stack imports;
  answer generation gated by `OPENROUTER_API_KEY`. Reuses `src/retrieval.py`, `src/llm_client.py`.
  Tabs: Overview, Table QA, Table Extraction, Layout, FUNSD Relations, Limitations.
- `notebooks/06_demo.ipynb` (PR-C), `notebooks/07_final_report.ipynb` (PR-B) — Colab runners.
- `reports/final_report.md` (PR-B) — methodology, metrics (generated-from-summary), GT-vs-OCR
  separation, limitations, future work, "reproduce in this order".
- `tests/test_phase4_summary.py` (new) — synthetic fixtures only (P3).
- Docs: `README.md` status refresh (no stale "Phase 2 active"); `DEVLOG.md` entry; `PLAN.md` §7.

## Out of scope (future work)
GriTS; Ragas / DeepEval; full-document (non-table) chunking; chart/figure extraction;
cross-encoder reranker / learned query routing; live PDF -> pipeline; HF Spaces deploy.

## Verification / gates
1. **Unit:** `pytest tests/test_phase4_summary.py` green, then full `pytest` green — synthetic,
   no Drive/network, no gradio.
2. **Summary build:** `python scripts/build_phase4_summary.py` writes the JSON + markdown; numbers
   match the sources (FUNSD test_50.qa F1 0.727; QA gt_markdown answer_exact 0.675; layout
   matched@0.50 recall 0.900).
3. **No-drift:** re-running the builder leaves `reports/phase4_metrics.md` byte-identical.
4. **Demo:** `scripts/run_demo.py` launches in the degraded case (no key, no embedding stack) and
   the full case.
5. **Report:** `reports/final_report.md` exists; README has no stale Phase 2 wording.

## Build order (TDD) + PR boundaries
- **PR-A (core, done):** tests -> `src/phase4_summary.py` -> `scripts/build_phase4_summary.py` ->
  generated `reports/phase4_metrics.md`; + README/DEVLOG/PLAN docs.
- **PR-B (report, done):** `reports/final_report.md` + `notebooks/07_final_report.ipynb`.
- **PR-C (demo, done):** `scripts/run_demo.py` + `notebooks/06_demo.ipynb`.

## Branch
`feature/phase4-demo` integrates PR-A/PR-B/PR-C and was cut from the latest `origin/main` after
`git fetch`.
