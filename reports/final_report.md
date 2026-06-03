# FinDocStructRAG — Final Report

A layout-aware pipeline for extracting structured tables from financial-report PDFs and
answering questions over them, plus a standalone form relation-linking baseline. This report
is the Phase 4 final report: it states what was built, how it was evaluated, and what the results
mean. **All metric numbers are generated** by `scripts/build_phase4_summary.py` into
`reports/phase4_metrics.md` and are never hand-copied into this prose;
`notebooks/07_final_report.ipynb` renders that generated table inline beneath this report.

## 1. What was built

- **Phase 1A — table topology.** Table Transformer (TATR) structure recognition is turned into
  a canonical row/column grid: spanning-cell mapping, grid validation, and occupancy-aware HTML
  parsing.
- **Phase 1B — content.** OCR (PaddleOCR) fills cell text; financial numbers are normalized.
  Ground-truth-filled and OCR-filled tables are kept **strictly separate** (project rule P4):
  GT-filled is a QA-validation reference only, never reported as an extraction output.
- **Phase 1C — table-only RAG.** One chunk per table; retrieval is BM25 + dense BGE cosine + RRF
  with **no LLM in the retrieval path** (P5). Answer generation is a single, swappable provider
  behind `src/llm_client.py`; the GT-filled and OCR-filled corpora are scored separately.
- **Phase 2 — layout.** A DocLayNet page-level detector finds table regions and crops them, which
  then feed the Phase 1A/1B pipeline (page -> crop -> structure -> content).
- **Phase 3 — FUNSD relations.** An annotation-only, deterministic question->answer linking
  baseline over ground-truth entities (per-answer argmax + distance gate), scored P/R/F1 against
  the GT links. No image pixels, no GPU.

Throughout, pretrained models are used for **inference and evaluation only** — nothing is
fine-tuned. The contribution is the layout-aware extraction + retrieval pipeline and an honest
measurement of it.

## 2. How it was evaluated

- **Subset evaluation, stated as such.** Metrics are computed on fixed, seeded subsets, not a
  whole-dataset benchmark. The goal is honest, reproducible numbers, not a leaderboard claim.
- **Deterministic / custom metrics.** Topology, content, retrieval, QA, layout, and relation
  metrics are all computed in-repo with explicit definitions (no GriTS / Ragas / DeepEval — those
  are future work, see §5).
- **GT vs OCR scored separately (P4).** The downstream answer-quality gap between a perfect
  (GT-filled) table and the real (OCR-filled) extraction is measured directly, not averaged away.
- **Honest held-out split for relations.** FUNSD heuristic parameters are set on `train_149`
  only; the headline is the held-out `test_50` (no tuning on the reported set).
- **Generated, not transcribed.** `scripts/build_phase4_summary.py` aggregates the per-phase
  artifacts into `outputs/evaluation/phase4_summary.json` and the committed
  `reports/phase4_metrics.md`. A no-drift check keeps the committed table byte-identical on
  rebuild, so the reported numbers cannot silently diverge from the artifacts.

## 3. Results

The full, generated metric tables live in **`reports/phase4_metrics.md`** (rendered inline by
`notebooks/07_final_report.ipynb`). Read qualitatively, the results show:

- **Table structure is strong; content is the harder half.** Column topology and cell-occupancy
  are recovered at high accuracy; OCR cell-text exactness is the lower number, as expected for a
  recognition step over scanned financial tables.
- **Retrieval is near-ceiling on linearized table chunks**, with sparse (BM25) competitive with
  or ahead of dense on this corpus, and RRF robust across corpora.
- **OCR introduces a measurable but bounded answer-quality cost.** On the QA set, the GT-filled
  corpus answers more exactly than the OCR-filled corpus — the gap is the price of the recognition
  step, and surfacing it is the point of the P4 separation.
- **Layout crops are accurate and hand off cleanly to TATR**, with a low false-positive rate on
  table-free pages and almost all crops passing the structure-validity smoke.
- **Relation linking is high-precision; recall is the design ceiling.** The single-link,
  geometry-only V1 recovers most question->answer pairs precisely but cannot, by construction,
  cover multi-answer or non-geometric links.

## 4. What this is not (limitations)

- **Subset, not full-corpus.** Numbers describe seeded subsets; they are not whole-dataset
  benchmarks.
- **GT-filled is a reference, not a result.** Per P4, GT-filled tables exist to validate the QA
  pipeline; the real extraction is OCR-filled, and the two are never conflated.
- **FUNSD V1 is annotation-only over GT entities.** It does not detect entities and does not read
  pixels; it is a relation baseline, not an end-to-end form parser. Single link per answer.
- **RAG is table-only.** Full-document (non-table) text is not chunked; charts and figures are not
  extracted (caption-level handling / future work).
- **The demo is artifact-backed.** It serves already-produced outputs and does live retrieval +
  answer generation over the existing chunks; it does not run a live PDF -> pipeline.

## 5. Future work

- **GriTS** for formal table-structure / content evaluation.
- **Ragas / DeepEval** for RAG faithfulness and answer-quality scoring.
- **Full-document chunking** beyond table-only RAG.
- **Chart / figure understanding** (chart-to-table, multimodal figures).
- **Cross-encoder reranker or learned query routing.**
- **FUNSD V2:** token classification (entity detection) and threshold-based multi-link matching.

## 6. Reproduce in this order

Data and the GPU-dependent runs happen on Colab; aggregation and the demo are local/CPU. The
per-phase notebooks (`notebooks/01`-`05`) are the runners for steps 1-6.

1. **Data.** Acquire FinTabNet.c and DocLayNet (notebooks `01` / `04` boot cells); fetch FUNSD
   with `python scripts/fetch_funsd.py`.
2. **Phase 1A topology.** `run_phase1a_colab.py` -> `evaluate_tables.py --run-id mvp_rand`.
3. **Phase 1B content.** `run_phase1b_gt_filled.py` + `run_phase1b_ocr_filled.py` ->
   `evaluate_content.py --run-id mvp_rand`.
4. **Phase 1C RAG.** `build_table_chunks.py` -> `build_qa_dataset.py` -> `evaluate_rag.py` ->
   `evaluate_qa.py`.
5. **Phase 2 layout.** `run_layout_batch.py` -> `eval_layout_iou.py --require-table-gt` (pos) and
   `--exclude-table-gt` (neg) -> `smoke_structure.py`.
6. **Phase 3 relations.** `evaluate_funsd.py`.
7. **Phase 4 summary.** `python scripts/build_phase4_summary.py` ->
   `reports/phase4_metrics.md` + `outputs/evaluation/phase4_summary.json` (this report reads the
   former).
8. **Demo.** `python scripts/run_demo.py` (key-optional Gradio final demo).
