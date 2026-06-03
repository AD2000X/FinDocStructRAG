# DEVLOG

A running engineering journal so a later reader sees *why* something works, not just
that it does. Two entry kinds:

- **Finding** (debugging) — `Symptom -> Error -> Root cause -> Fix -> Commit`. Dated,
  chronological, newest on top.
- **Decision** (settled engineering choice) — `Decision -> Rationale -> Alternatives ->
  Impact -> Files/Commits`. Cross-cutting; kept in the Decisions section below.

Record only settled decisions and concrete fixes; do not transcribe the chat. If
Decisions outgrow this file, split them into `DECISIONS.md` (or `docs/adr/`).

---

## Engineering decisions

### Decision - Inference + evaluation pipeline, not model retraining

- **Decision:** Phase 1A (and the OCR/layout phases) run pretrained models (TATR,
  PaddleOCR, DocLayNet, BGE embeddings) for inference and evaluation only; we do not
  fine-tune them.
- **Rationale:** the project's value is a layout-aware extraction + retrieval pipeline,
  not a new model. Under Colab T4 limits, fine-tuning is out of scope and unnecessary.
- **Alternatives:** fine-tune TATR on FinTabNet.c (rejected: cost/time, no clear gain
  for the deliverable).
- **Impact:** no train split needed; the discipline that matters is a fixed, honest
  *evaluation* split (see next decision), not training-leakage control.
- **Files/Commits:** project-wide; PLAN.md phases.

### Decision - Fixed evaluation subsets instead of train/val/test

- **Decision:** evaluate on fixed debug/mvp/final subsets drawn by seed, not a training
  split.
- **Rationale:** we infer only, so the real risks are non-representative sampling and
  tuning on the held-out set - not training leakage.
- **Convention:** debug = seed 7 / limit 50; mvp = seed 42 / limit 300; final =
  seed 2026 / limit 500-1000 (Colab time permitting). final is run last and never used
  to tune thresholds.
- **Alternatives:** classic train/val/test (rejected: no training); a single ad-hoc
  batch (rejected: not reproducible, easy to over-tune).
- **Impact:** every reported metric must carry subset name, seed,
  processed/skipped/failed, and "metrics over successful samples only".
- **Files/Commits:** DESIGN_SPEC 18.6; `find_xml_files(seed=...)`; `77e422d`.

### Decision - Seeded random sampling (seed 42) over first-N

- **Decision:** draw the mvp sample with `--seed 42`, not the first 300 sorted files.
- **Rationale:** `find_xml_files` sorts by path, so "first 300" is alphabetically-first
  and in practice almost all one issuer (ADS_2007/2008). Numbers on it describe that
  issuer, not FinTabNet.c. A seeded shuffle samples across issuers/years.
- **Alternatives:** first-N (rejected: issuer-biased); `random.sample(files, limit)`
  (rejected: not nested across limits, breaks resume when the limit grows). We use
  shuffle-then-slice so seed 42's 10 ⊂ 50 ⊂ 300.
- **Impact:** mvp run-id changed from the ADS-only `mvp` to `mvp_rand`; the ADS numbers
  (row 0.897 etc.) are explicitly a non-representative subset, superseded by mvp_rand.
- **Files/Commits:** `find_xml_files`; runner `--seed`; `77e422d`.

### Decision - Add tatr_raw/ for accurate prediction visualisation

- **Decision:** persist a raw TATR artifact per sample (`outputs/tables/tatr_raw/`)
  alongside the canonical `tatr_predicted/`.
- **Rationale:** the canonical grid drops raw boxes/scores/labels and the header classes,
  but deliverable screenshots #2 (TATR box overlay) and #5 (grid-geometry report) need
  them. Storing GT and prediction in separate streams also keeps a "GT annotation
  overlay" from being mislabelled as a "TATR prediction overlay" (P4).
- **Alternatives:** re-run the GPU model when drawing figures (rejected: wasteful, and
  the run would have to be repeated for every figure pass); draw predicted boxes from the
  GT XML (rejected: that is ground truth, not prediction - a correctness/honesty bug).
- **Impact:** one extra small JSON per sample on Drive; #2-#5 become drawable on CPU
  later. Built by the pure, tested `src/tatr_raw.py`.
- **Files/Commits:** `src/tatr_raw.py`; DESIGN_SPEC 18.5; `77e422d`.

### Decision - gt_filled text source + conservative word-assignment guard (Phase 1B)

- **Decision:** gt_filled cell text is built by assigning FinTabNet.c word-level GT
  tokens (`words/<stem>_words.json`) to the GT structure grid via the same
  `assign_words_to_cells` used for OCR. The matching rule is center-in-cell -> max IoU
  -> nearest row x nearest col, but only inside a conservative expanded-grid guard
  (margin = `max(pct*extent, one median row-height / col-width)`); otherwise the word is
  left unassigned. The guard is NOT widened to swallow lines that sit a full row or more
  outside the grid.
- **Rationale:** GT topology and OCR topology then flow through one identical fill path,
  so content metrics isolate (TATR grid vs GT grid) and (OCR text vs GT text) rather than
  mixing in a second cell-string parser. The guard recovers words whose bbox edges just
  miss the grid (a few px) without forcing genuine out-of-grid text (captions, unit
  labels, subtotal headers, page residue) into the nearest cell. Confirmed concretely on
  AAL_2002_page_41_table_1: a "Total $729 $506 $386" line sits at y 0.75-12.39 while the
  grid starts at y 29.37 - a ~17px gap, ~1 median row-height - so it stays unassigned by
  design. Phase 1B measures content reconstruction inside the GT grid, not full-crop word
  recall.
- **Alternatives:** official FinTabNet cell-level HTML strings from the PDF_Annotations
  archive (deferred to future work: needs a second archive + HTML/grid alignment, and
  bypasses the shared assignment path); widen the guard to catch the floating header
  (rejected: pollutes other samples with caption/footer text); nearest-cell fallback
  instead of nearest row x col (rejected: less faithful to row/column structure).
- **Impact:** gt_filled and ocr_filled share `fill_table`; unassigned words are tracked
  and word-assignment coverage is reported per run. Documented as a limitation
  (DESIGN_SPEC 14) and a coverage-reporting rule (DESIGN_SPEC 5.12, 6.2). gt_filled text
  is reconstructed GT-words-on-GT-grid, not the official cell HTML string - stated in the
  report so it is not over-read.
- **Files/Commits:** `src/ocr_adapter.py`, `src/table_fill.py`,
  `src/tatr_postprocess.py` (`assign_words_to_cells`), `src/fintabnet_loader.py`
  (`parse_words_json`/`words_path_for`), `scripts/run_phase1b_gt_filled.py`,
  `scripts/preview_table.py`; DESIGN_SPEC 5.12/6.2/14; `ee7ae4f`..`f158796`.

### Decision - Phase 1B content metrics are multi-view proxy metrics, not pure OCR accuracy

- **Decision:** report Phase 1B content quality as three spatial views, not one number,
  and label them transparent proxies (not a full GriTS/TEDS benchmark):
  - **one_to_one (strict):** each GT cell vs its single max-IoU pred cell; penalises
    topology fragmentation.
  - **aggregate (primary):** each GT cell gathers every pred cell centered inside it,
    joins text in reading order, then compares; measures content recovery within the GT
    cell region, robust to TATR over-segmentation.
  - **topology_matched_subset:** aggregate metric restricted to samples whose row/col
    counts equal GT; isolates OCR + assignment quality from topology errors.
- **Rationale:** ocr_filled is the whole extraction chain (TATR topology + PaddleOCR
  recognition + word-to-cell assignment + reconstruction), so a low cell_text_exact_match
  does NOT mean "PaddleOCR is 20% accurate" - the loss can be over-segmentation, cell
  misalignment, one GT cell split across pred cells, OCR errors, assignment errors, or GT
  words outside the GT grid. A single number hides which. Two bugs/decisions surfaced
  this: (1) aligning cells by (row_start,col_start) INDEX was a real bug - TATR
  over/under-segmentation shifts indices so it compared physically different cells
  (mean_alignment_iou ~0.86 after switching to spatial bbox-IoU confirmed the fix); (2)
  even with correct alignment, 1-to-1 matching scores a split cell as a miss - that is a
  measurement-design choice, not a bug, so it is reported alongside aggregate rather than
  tuned away. Principle held to: metrics are defined by the question they answer, decided
  before judging whether the numbers look good; report multiple views, never cherry-pick
  the highest.
- **Alternatives:** single index-aligned exact-match number (rejected: the index bug, and
  it conflates OCR with topology); only the aggregate number (rejected: hides the
  fragmentation cost); implement full GriTS/TEDS now (deferred: heavy 2D DP alignment;
  our aggregate mode is a simplified GriTS-Con proxy). GriTS/TEDS remain the rigorous
  standard and stay listed as future work (DESIGN_SPEC 6.1).
- **Impact:** evaluate_content.py emits all three views; the gap between one_to_one and
  aggregate localises topology-vs-OCR loss. Honest framing for the report: "transparent
  proxy metrics; full GriTS/TEDS is future work."
- **Files/Commits:** `src/eval_content.py`, `scripts/evaluate_content.py`,
  `tests/test_eval_content.py`; DESIGN_SPEC 6.1/6.2/14; `c098255`, `c727f3f`.

### Decision - Dataset cache on /content, outputs on Google Drive

- **Decision:** extract the FinTabNet.c archive to Colab scratch (`/content/...`); write
  pipeline outputs (manifests/metrics/failures/tables/figures) to Drive.
- **Rationale:** the archive is cheaply re-downloadable from HF and extracting thousands
  of small files to Drive is very slow; only the expensive, hard-to-reproduce *outputs*
  need to survive a dropped session.
- **Alternatives:** everything on Drive (rejected: slow extraction, Drive clutter);
  everything on scratch (rejected: outputs lost when the VM dies, breaking resume).
- **Impact:** outputs are gitignored and persist to Drive; the dataset cache is
  disposable. `config` switches roots automatically on Colab vs local.
- **Files/Commits:** `src/config.py`; `fintabnet_loader._dataset_cache_root`; DESIGN_SPEC 18.1.

### Decision - Decouple evaluation from the GPU runner

- **Decision:** the GPU runner produces artifacts; `scripts/evaluate_tables.py`
  (CPU-only) computes the authoritative report from all persisted predictions.
- **Rationale:** a resumed/partial run only "sees" what it processed, so a coupled report
  silently covers a subset (a full skip even clobbered a good report with zeros). Scoring
  over every persisted prediction is correct regardless of how many sessions produced it.
- **Alternatives:** report inside the runner (rejected: subset/clobber bug, and it needs
  the GPU just to re-score).
- **Impact:** metrics are reproducible on CPU and stable across resumes. See the dated
  Finding below for the bug that triggered this.
- **Files/Commits:** `46b4dbc`, `bd242c6`.

### Decision - TATR processor: use_fast=False + injected shortest_edge

- **Decision:** load the structure model's image processor with `use_fast=False` and set
  `processor.size = {shortest_edge: min(800, longest), longest_edge: longest}`.
- **Rationale:** the fast DETR processor crashes in post-processing on this transformers
  version (`'SizeDict' object has no attribute 'keys'`); and the checkpoint ships a
  longest-edge-only size, which the resize step rejects. Both are environment/checkpoint
  quirks, not pipeline bugs.
- **Alternatives:** pin an older transformers (rejected: heavier, fights the Colab
  stack); height/width resize (rejected: loses TATR's MaxResize aspect behavior).
- **Impact:** stable CPU/GPU inference; predicted boxes stay in original-crop
  coordinates. Details in Issues 2 and 3 below.
- **Files/Commits:** `6b71082`, `1e6c7e5`.

---

## 2026-06-03 - Phase 3 FUNSD relation-linking baseline (V1)

### Result - annotation-only spatial heuristic; high precision, recall is the design ceiling

First Phase 3 deliverable: a deterministic FUNSD relation-linking baseline over GT entities,
CPU-only and annotation-only (the FUNSD JSON carries entity text/bbox/label and the GT
`linking` pairs, so no image pixels are loaded). Run on the real dataset (149 train + 50 test
= 199 forms), `scripts/evaluate_funsd.py`, untuned a-priori params:

| split | scope | precision | recall | f1 | tp / pred / gold | n |
| --- | --- | --- | --- | --- | --- | --- |
| **test_50** | **qa_links** | **0.946** | **0.590** | **0.727** | 494 / 522 / 837 | 50 |
| all_199 | qa_links | 0.925 | 0.535 | 0.678 | 2123 / 2295 / 3966 | 199 |
| test_50 | all_links | 0.946 | 0.464 | 0.623 | 494 / 522 / 1064 | 50 |
| all_199 | all_links | 0.925 | 0.401 | 0.560 | 2123 / 2295 / 5293 | 199 |
| train_149 | qa_links | 0.919 | 0.521 | 0.665 | 1629 / 1773 / 3129 | 149 |

Reading it honestly:
- **Headline (held-out): `test_50.qa_links.micro_f1` = 0.727**, precision 0.946. The heuristic
  fires conservatively and is right when it does; the limit is recall.
- **Recall (0.590) is the design ceiling, not a bug.** Per-answer argmax emits at most one link
  per answer, and the geometry only models same-row right-side and below relations - so answers
  whose question sits left/above, or that have multiple gold questions, are under-covered. The
  rejected alternatives (per-question argmax, global threshold) trade this for precision; richer
  matching (threshold-based multi-link) is the documented next lever, deliberately out of V1.
- **No tuning-on-test risk.** Params are a-priori defaults; `train_149` F1 (0.665) is *below*
  `test_50` (0.727), so test is if anything the easier split - the gap is sampling, not fitting.
- **`all_links` is a coverage diagnostic, not a second predictor.** Same QA predictions scored
  (as undirected pairs) against every GT link; recall necessarily drops (0.464 test) because the
  QA-only heuristic cannot cover header->question and other link types. `all_199` carries the
  "contains the 50 test + 149 tuned forms, not held-out" caveat in the report JSON.

Design (locked in discussion; see `docs/phase3_brief.md`):
- **Predictor:** per-answer argmax + distance gate; distances normalized by the form's median
  entity height; two separate knobs (`max_distance_units` distance gate, `min_score` floor).
- **GT links:** deduped to undirected frozensets (FUNSD records links bidirectionally), then
  `qa_gold_links` canonicalizes question+answer to directed `(q,a)`; `all_gold_links` keeps the
  full undirected set.
- **Reporting matrix:** primary `test_50.qa_links` (held-out); secondary `all_199.qa_links`,
  `test_50/all_199.all_links`; `train_149` is the dev/tuning split.
- **No sklearn in V1** (P/R/F1 is set arithmetic). **No image loading** (optional later for
  qualitative overlay/debug only, never in the baseline or the gate).
- **Scope held:** standalone branch; does not touch the RAG pipeline. FUNSD token classification
  (V2 / seqeval) is future work.
- **Files:** `src/funsd_extraction.py`, `src/eval_funsd.py`, `scripts/evaluate_funsd.py`,
  `scripts/fetch_funsd.py`, `tests/test_funsd_relations.py` (17 synthetic tests, the gate),
  `src/config.py` (FUNSD paths). Full suite 236 passed.

---

## 2026-06-02 - Phase 2 DocLayNet layout-crop MVP gate

### Finding - Aryn primary carries forward; fallback is narrow; crop->structure needs band dedup

- **Symptom:** the first debug layout batch made fallback look useful and produced noisy
  structure-smoke WARN rows. The numbers were misleading because fallback fired on table-free
  pages and the TATR structure normalizer kept overlapping row/column bands.
- **Root cause:** two separate issues:
  1. `detect_layout` treated "primary found zero tables" as a fallback trigger, but
     `microsoft/table-transformer-detection` generates many false positives on table-free
     DocLayNet pages.
  2. Structure recognition often emitted overlapping adjacent row/column bands; the geometry
     validator correctly warned, but the rows/cols needed 1-D NMS before grid normalization.
- **Fix / decision:** carry forward `Aryn/deformable-detr-DocLayNet` as the primary detector
  with `table_threshold=0.30` and `dedup_iou=0.70`; skip fallback when primary detects zero
  tables; make `dedup_row_col_bands` the default before `normalize_tatr_prediction`.
- **Evidence:** fixed DocLayNet MVP subset (seed 42, n=200) scored mean best table-crop IoU
  0.900 on GT-table pages; table-level matched@0.50 recall 0.900 / precision 0.916 and
  matched@0.75 recall 0.880 / precision 0.895. On 200 table-free pages, final crop false
  positives were 13/200 (6.5%) and fallback fired 0/200.
- **Structure handoff:** n=50 crop smoke improved from 37 OK / 13 WARN before band dedup to
  50 OK / 0 WARN after band dedup. Step 7d full-crop smoke confirmed (seed=42, n=286):
  **285 OK / 1 WARN** (0.35%); the sole WARN is `val_000670_table_1` (rows=0, no row boxes
  detected), well under the <=5% WARN gate.
- **Scope caveat:** these are fixed-subset Phase 2 diagnostics, not whole-DocLayNet AP. The
  crop->TATR smoke validates grid geometry compatibility, not OCR text/content quality.
- **Files/Commits:** `src/bbox_utils.py`, `src/layout_parsing.py`, `src/layout_detector.py`,
  `src/table_detection.py`, `scripts/run_layout_batch.py`, `scripts/eval_layout_iou.py`,
  `scripts/smoke_structure.py`, `notebooks/04_phase2_layout.ipynb`; commits `6588e02`,
  `0d94518`, `7652d22`, `0eb0ba9`, `4863f3f`.

## 2026-06-01 - Phase 1C manual QA answer eval (temperature=0, locked matrix)

### Finding - linearized wins reproducibly, abstain is perfect, the real gap is one subset-sum

Added the 16-row hand-authored manual set (10 numeric reasoning + 6 unanswerable) to the QA
set (n=40 answerable + 6 unanswerable) and re-ran the full answer eval. Two reruns first
jittered 1-3 questions per config (e.g. gt_linearized numeric_relaxed 0.825 -> 0.800 while
mq_0003 was *fixed*), because answer generation used the provider default temperature (~1.0).
Set `temperature=0` (greedy) in `build_openrouter_complete` so the matrix is a reproducible
measurement, then locked these numbers. numeric_relaxed is the headline (answer_exact is
depressed only because several golds carry a `$` the model omits):

| config | answer_exact | numeric_relaxed | citation_hit | abstain_accuracy |
|---|---|---|---|---|
| gt_markdown | 0.675 | 0.775 | 0.800 | 1.000 |
| gt_linearized | 0.650 | **0.875** | 0.825 | 1.000 |
| ocr_markdown | 0.550 | 0.700 | 0.750 | 1.000 |
| ocr_linearized | 0.575 | 0.800 | 0.850 | 1.000 |

Three conclusions, all reproducible:

- **linearized > markdown by a clean +0.10 numeric_relaxed on both corpora** (gt 0.875 vs
  0.775; ocr 0.800 vs 0.700). With decoding fixed this is signal, not sampling. linearized is
  the carry-forward serialization. OCR tax is a steady ~0.075 (3 questions) on both.
- **abstain_accuracy = 1.000 (6/6) on all four corpora.** Every plausible-but-unanswerable
  question is refused, including the one (uq_0002, APH remaining-useful-life) with mild
  cross-table risk. Grounding/abstention is the strongest result in the phase.
- **The remaining answerable weakness is multi-row arithmetic, not retrieval / citation /
  verbosity.** Every model answer is a bare number or empty string (no chatty
  "$285 (495-210)" that would break the numeric matcher). Of the 10 manual rows, 8 are
  correct; the two misses are mq_0009 (`9.1` vs gold `11.7` - a subset-sum: "sum only the
  positive-fair-value rows" - the genuine, durable failure) and mq_0007 (`7.9` vs `8.0`, a
  borderline cell-read/rounding slip just over the 1% tolerance).

Determinism mattered for honesty: mq_0010 (a large subtraction, gold 7,382,771) answered a
wrong 724,771 under stochastic decoding but the correct 7,382,771 at temperature=0, and
mq_0003's percentage-point gold was re-authored to forbid a `%` sign (the relaxed matcher
treats `31.0%` as ratio 0.31). So the earlier "model fails big-number subtraction" read was
partly decoding noise; the durable arithmetic limit is the single subset-sum question.

Caveats: n=40 answerable is small, and the templated lookups dominate it, so treat the manual
subscore (8/10) as directional. answer_exact undercounts by the currency-symbol formatting
(numeric_relaxed is the metric to read). The manual set is what turned this from a plumbing
check (see the templated-baseline entry below) into a measurement with real abstain signal.

## 2026-06-01 - Phase 1C answer-gen full matrix (clean templated baseline)

### Finding - linearized carried forward; OCR adds a modest answer penalty; weakness is citation precision

Full Step 5 over all four corpora (RRF retrieval top-k=10 -> `openai/gpt-4o-mini` grounded
answer, 4 x 30 = 120 calls). Confirmed gate-clean before reading: VM HEAD = `8e0e8d6` (the
generator quality gate) and `outputs/qa/qa_all.jsonl` = 30 lines, i.e. the QA set was
regenerated through the gate and carries no manual seed, so the within-table-ambiguity noise
(tq_0001 class) is excluded and abstain_rate is a pure false-abstain diagnostic. numeric_relaxed
is the headline (it strips `13,223` vs `13223` formatting noise that depresses answer_exact on
numeric lookups):

| config | answer_exact | numeric_relaxed | citation_hit | abstain_rate |
|---|---|---|---|---|
| gt_markdown | 0.700 | 0.733 | 0.700 | 0.067 |
| gt_linearized | 0.767 | **0.900** | 0.733 | 0.000 |
| ocr_markdown | 0.600 | 0.633 | 0.600 | 0.067 |
| ocr_linearized | 0.633 | **0.800** | 0.767 | 0.067 |

Phase 1C templated QA baseline, n=30:

- **Linearized serialization carries forward.** linearized beats markdown by an identical
  +0.167 numeric_relaxed on both corpora (gt 0.733->0.900, ocr 0.633->0.800). Strong baseline
  signal for the engineering decision; not over-claimed (n=30, templated).
- **GT is an oracle upper bound, not the pipeline winner.** The end-to-end OCR pipeline is
  represented by **ocr_linearized (numeric_relaxed 0.800, citation_hit 0.767)**; gt_linearized
  (0.900) is the headers-and-values-from-GT ceiling.
- **OCR downstream tax: ~10 points numeric_relaxed** (a steady 0.100 gap on both
  serializations). Retrieval recall@10 was 1.000 for both corpora, so OCR is not losing the
  table - it misreads the cell value once the right table is in context.
- **Citation precision remains weaker than retrieval recall.** citation_hit 0.60-0.77 while RRF
  recall@10 = 1.000: the gold chunk is always in the top-10, but the LLM does not stably emit
  the gold chunk id in a multi-chunk context. The bottleneck is LLM citation/grounding emission,
  not chunk retrieval. Recorded; not fixed now.
- **Manual/unanswerable QA still needed before interpreting abstain behavior.** All 30 are
  answerable, so abstain_rate only counts false abstains; abstain ability is unmeasured until
  the hand-authored unanswerable set is added (Track B). Do not read answer_exact as final RAG
  accuracy.

(HF "unauthenticated" warning and the BGE `embeddings.position_ids UNEXPECTED` load message are
both benign and ignored.)

## 2026-06-01 - Phase 1C answer-gen smoke + QA generator quality gate

### Finding - templated lookups can be within-table ambiguous; added a generator gate

The first OpenRouter answer-gen smoke (`openai/gpt-4o-mini`, gt_markdown, 3 questions) scored
answer_exact 0.333 / numeric_relaxed 0.333 / **citation_hit 1.000** / abstain 0.000. The
citation_hit = 1.0 is the signal: retrieval + grounding work (the model cites the right table
every time, including one retrieved at rank 2). The two misses were cell selection / question
quality, not retrieval:

- **tq_0003** (gold `(17)`, answer `178`, correct table): grid cell mis-selection - the model
  had to align a bare value to row x column by counting markdown pipes and grabbed a neighbor.
  This is a serialization weakness; linearized (value paired with its header) should fix it.
- **tq_0001** (gold `1,003`, answer `$4.61`, correct table): the table has two body rows
  labelled "Diluted" (Diluted EPS = $4.61, Diluted shares = 1,003), so "What was Diluted in
  2010?" is ambiguous and the generator's gold is arbitrary. This is a **broken question**, not
  a model error - and serialization-invariant (linearized keeps both "Diluted: ..." lines too).

So the two error classes are different and only one is a serialization issue. Fix for the
second: a generator quality gate (`qa_templates._usable_row_label` + a within-table label
Counter) that skips a lookup whose row label is non-unique among the body rows, or too short /
non-alphabetic (years, footnote markers). Folding the section/parent label into the question
to disambiguate rather than skip is deferred. The 30-question set is sampled across 286 tables,
so the gate does not starve it.

Reading rule recorded for the full run: judge the three deltas, not the absolute score -
(a) linearized > markdown? (b) gt ~= ocr? (c) citation_hit stays high? - because templated
answer_exact carries an ambiguity noise floor and is a plumbing + serialization-tiebreak
measure, not the final RAG accuracy (that needs the manual + unanswerable set).

## 2026-06-01 - Phase 1C dense + RRF retrieval method matrix

### Result - BM25 vs dense (BGE) vs RRF over the 4 corpora (mvp_rand 300, 30 templated QA)

Added the dense path (`bge-small-en-v1.5`, exact cosine) and RRF fusion of BM25+dense, all
behind the same `query -> ranked chunk_ids` contract. Full method x corpus matrix:

| corpus | method | hit@1 | hit@5 | hit@10 | mrr@10 |
| --- | --- | --- | --- | --- | --- |
| gt_markdown | bm25 | 0.933 | 0.967 | 0.967 | 0.950 |
| gt_markdown | dense | 0.667 | 0.800 | 0.900 | 0.715 |
| gt_markdown | rrf | 0.833 | 1.000 | 1.000 | 0.917 |
| gt_linearized | bm25 | 0.767 | 1.000 | 1.000 | 0.864 |
| gt_linearized | dense | 0.800 | 0.900 | 0.933 | 0.845 |
| gt_linearized | rrf | 0.867 | 0.967 | 1.000 | 0.906 |
| ocr_markdown | bm25 | 0.933 | 0.967 | 0.967 | 0.950 |
| ocr_markdown | dense | 0.600 | 0.833 | 0.933 | 0.683 |
| ocr_markdown | rrf | 0.833 | 0.967 | 1.000 | 0.894 |
| ocr_linearized | bm25 | 0.733 | 1.000 | 1.000 | 0.842 |
| ocr_linearized | dense | 0.767 | 0.933 | 0.933 | 0.816 |
| ocr_linearized | rrf | 0.833 | 0.933 | 1.000 | 0.887 |

Reading it (n=30, all templated *lexical* lookups - this corner of the space structurally
favors BM25):
- **Dense alone < BM25 on these lookups, not a bug.** The questions hinge on exact row-label
  + year tokens (BM25's home turf), and bge-small compresses ~300 near-identical financial
  tables into vectors that do not discriminate precise lookups well. Plausible range, and the
  pipeline behaves sensibly (RRF fusion, serialization split below) - not an embedding bug.
- **Serialization interacts with retriever type (citable).** Dense does clearly better on
  linearized than markdown (hit@1 0.800/0.767 vs 0.667/0.600): markdown is `|`/`---`/number
  -soup that BGE never trained on, while linearized reads like prose. So markdown favors
  lexical (BM25), linearized favors dense. (Secondary hypothesis: large markdown tables may
  exceed BGE's 512-token limit and truncate; linearized is more compact.)
- **RRF is the recall winner: hit@10 = 1.000 on all four corpora**, and it fixed markdown's
  one hard miss (gt_markdown hit@5/@10 0.967 -> 1.000). The table BM25-markdown could never
  surface, dense ranked within top-10, so fusion pulled it in - the RRF value proposition.
- **RRF trades top-1 on markdown** (0.933 -> 0.833: weak dense dilutes BM25's strong #1) but
  *lifts* it on linearized (0.767 -> 0.867: dense is competitive there, so fusion helps rank-1).
- **OCR vs GT, now fair across all three methods: ~0-1 question gap everywhere** (largest, still
  tiny, on dense+markdown 0.667 -> 0.600 - OCR noise hurts a single semantic vector most).

Decisions (settled for now):
- **Method: carry RRF forward.** For table-RAG that feeds top-k to an LLM, "is the answer table
  in the context" (recall@k) matters more than top-1; RRF gives perfect hit@10 everywhere.
- **Serialization: do NOT pick a winner on retrieval.** markdown+rrf and linearized+rrf are
  within ~1 question; the tie-break is which format the LLM reads more accurately -> defer to
  the answer-generation eval.
- **Do not write off dense.** Templated lexical questions are its worst case; the manual /
  paraphrased set is dense's main stage, and RRF hedges against the lexical bias.
- **OCR impact:** "OCR does not materially degrade table retrieval on this subset" is now
  supportable across bm25 / dense / rrf.
- **Commits:** `8acbff9` (dense + RRF), `3e065d8` (--top-k guard).

## 2026-06-01 - Phase 1C table-only retrieval BM25 baseline (fair)

### Result - first fair 4-corpus BM25 retrieval baseline (mvp_rand 300, 30 templated QA)

Table-only RAG retrieval over the 4 corpora ({gt,ocr} x {markdown,linearized}), 300 chunks
each, scored with 30 templated-from-GT lookup questions (one relevant table each). BM25 only,
no LLM, no API key (P5). Metrics over answerable questions (`scripts/evaluate_rag.py`).

| corpus | hit@1 | hit@5 | hit@10 | mrr@10 |
| --- | --- | --- | --- | --- |
| gt_markdown | 0.933 | 0.967 | 0.967 | 0.950 |
| ocr_markdown | 0.933 | 0.967 | 0.967 | 0.950 |
| gt_linearized | 0.767 | 1.000 | 1.000 | 0.864 |
| ocr_linearized | 0.733 | 1.000 | 1.000 | 0.842 |

Reading it honestly (n=30, all templated - a plumbing baseline, not the final QA number):
- **OCR barely hurts retrieval.** markdown: gt == ocr exactly (markdown ignores is_header, so
  both render identically; OCR text is good enough that BM25 matches the same tokens).
  linearized: gt 0.767 vs ocr 0.733 at hit@1 = 23/30 vs 22/30, one question, both recall@5 =
  1.000 -> indistinguishable at this sample size. Clean headline: BM25 table retrieval is
  robust to our OCR error level.
- **markdown vs linearized is a precision/recall tradeoff, consistent across gt and ocr.**
  markdown wins rank-1 (hit@1 ~0.93: headers stated once -> more discriminative top hit) but
  has one table it never surfaces (hit@10 0.967). linearized wins recall (hit@5/10 = 1.000,
  no misses) at lower hit@1 (~0.75): per-row header repetition dilutes rank-1 precision but
  guarantees a query term matches somewhere. Same mechanism both directions; neither dominates.
- **Cross-table ambiguity is the documented reason linearized shows hit@1 < hit@5** (right
  table in top-5 but not #1): many tables share generic year/date headers. The manual QA set
  (harder + unanswerable) is what will disambiguate; it is a later step, not a blocker.

Decision: do NOT pick a serialization winner yet - the recall/precision split is exactly why
both are carried into the dense + RRF and answer-generation stages. Dense (bge+FAISS) + RRF is
the next slice, to test whether dense fixes markdown's one hard miss and whether RRF(BM25,dense)
gets markdown's hit@1 with linearized's recall@5.

### Finding - the earlier "ocr_linearized > gt_linearized" was a header-asymmetry artifact

- **Symptom:** first retrieval run had ocr_linearized hit@1 = 0.900 *above* gt_linearized
  0.767 - reading as if OCR retrieved better than GT, which is implausible.
- **Root cause:** gt_filled was regenerated with column-header marking (is_header), but
  ocr_filled was built from tatr_predicted and had no headers. So ocr_linearized was the
  compact `label: val; val` form while gt_linearized carried diluting repeated headers - two
  different serializations, not a GT-vs-OCR text comparison. The shorter ocr text scored
  *better* under BM25 length normalization, inverting the apparent ranking.
- **Fix:** `scripts/mark_ocr_filled_headers.py` reads `column_headers` from `tatr_raw/<id>.json`
  (same TATR coordinate space as the predicted grid) and applies the same IoMin marking
  (`_mark_column_headers`) used for gt_filled - a fairness fix, no OCR re-run. patched 285/300;
  15 have no header band (genuinely header-less tables, e.g. maturity schedules); 0 missing
  tatr_raw. After rebuilding the OCR corpora, ocr_linearized hit@1 dropped 0.900 -> 0.733,
  confirming the inversion was entirely the artifact. Commit `29b3476`.
- **Diagnostic that caught it:** the re-run metrics were byte-identical to the pre-patch run.
  Identical scores across all 6 columns meant the ocr_linearized corpus text had not changed -
  Step 1 (build_table_chunks) must be re-run *after* the patch and *before* re-scoring.
- **Lesson:** any cross-source comparison must hold the serialization fixed; is_header silently
  changes what `linearized` emits. Also: when a "fixed" run reproduces identical numbers,
  suspect a stale artifact, not a confirmed result.
- **Follow-up (not done, flagged):** `run_phase1b_ocr_filled.py` itself still does not mark
  headers, so a fresh OCR re-run would re-introduce the gap; it should source headers from
  tatr_raw the same way when the final GPU OCR run happens.

## 2026-06-01 - Phase 1B content mvp_rand 300 (milestone)

### Result - Phase 1B content extraction (representative random subset, seed 42, 300)

The Phase 1B deliverable, on the seed-42 / 300-sample subset (run-id `mvp_rand`), after the
word-level OCR + rejoin + clean-join chain. Three views (DESIGN_SPEC 6.2); content metrics
over spatially-aligned cells only.

| view | n | align_cov | exact_match | numeric_relaxed | non_empty_f1 |
| --- | --- | --- | --- | --- | --- |
| aggregate (many-to-one) | 300 | 0.990 | 0.804 | 0.876 | 0.977 |
| one-to-one (strict, IoU>=0.5) | 300 | 0.973 | 0.761 | 0.826 | 0.906 |
| topology-matched subset | 234 | 0.999 | 0.819 | 0.902 | 0.988 |

Reading it honestly:
- **Coverage held ~0.99** across all views, so the numbers are real signal, not a
  coverage artifact - the worry when a metric jumps. (aggregate `gt_cells` 16184,
  `matched_cells` 16021.)
- **234/300 (78%) are topology-matched** (TATR grid count == GT), and on that clean subset
  numeric relaxed is **0.902** and f1 **0.988** - i.e. given a correct grid, OCR content
  reconstruction is strong. The aggregate/topology gap is dominated by TATR topology, not OCR.
- Numbers are *higher* than the 10-sample `debug_clean` (aggregate numeric 0.788 -> 0.876,
  exact 0.684 -> 0.804): the larger sample spans more clean numeric tables and dilutes the
  few hard text-heavy ones from the debug set. Expected, not suspicious.
- These are transparent PROXY metrics; TEDS / GriTS-Con remain the rigorous standard (future
  work). GT-filled is QA validation only and is not reported as an extraction output (P4).

Run on CPU PaddleOCR (the verified path). GPU paddle is a non-blocking parity smoke
(`gpu_smoke` vs frozen `debug_clean`); same models, so it does not change this milestone.

## 2026-05-31 - Phase 1B content error analysis

### Finding - word-level OCR boxes + clean join resolve both failure classes

Follow-up to the two-failure-types finding below. Both classes traced to PaddleOCR 3.x
emitting **line/phrase-level** detection boxes that straddle adjacent narrow financial
columns (confirmed on the IP/MA overlays: a single red box covered two GT columns).

- **Fix 1 - geometry.** Build `PaddleOCR(..., return_word_box=True)` and prefer the
  word-level `text_word` / `text_word_region` pair in `_parse_v3` (whitespace tokens
  dropped; the only score is per line, shared across its words). Probe-confirmed schema
  on Colab before coding. Result: each token now lands in its own column; IP_2012 went
  to 28/28 cells and the `$10,376 $ 9,812` merges disappeared. Commit `cd905b6`.
- **Fix 2 - numeric formatting.** Word split puts spaces around a number's separators
  ("13 , 223"). `normalize_financial_number` re-joins digit groups split *only* by a
  separator (`(\d)\s*([,.])\s*(\d)`), so a space NOT flanking a separator ("2011 2010")
  is left intact and genuine merges stay rejected. Commit `43c7078`.
- **Fix 3 - text formatting + the data product.** A naive space-join also dirties text
  cells ("Management ' s", "( Unaudited )"), which would degrade the Phase 1C RAG chunk,
  not just the metric. `join_word_tokens` applies conservative spacing (no space before
  closing punctuation / separators / % / apostrophe; no space after currency / opening
  brackets; "' s" contracts) and is shared by gt_filled and ocr_filled so the comparison
  stays symmetric. Raw tokens stay in `cell["words"]`. It does **not** change characters:
  a comma misread as a period ("29.2018") stays a visible mismatch. Commit `b0faa39`.

Verified on the seed-42 / 10-sample subset (run-id `debug_clean`), aggregate / one-to-one
/ topology-matched:

| metric | line-level (pre-fix) | word-level + clean join |
| --- | --- | --- |
| numeric_cell_relaxed_match | 0.373 / - / 0.481 | **0.788 / 0.738 / 0.772** |
| cell_text_exact_match | 0.236 / - / 0.299 | **0.684 / 0.647 / 0.617** |

Every remaining diff is a genuine OCR/detection error (`fiscal vears`, `29.2018` comma
misread, truncated `Accident`, `—`/`$ —` detection misses) or a benign currency-symbol
position (`158,389 $`, numerically correct -> NUM_OK). No spacing-only artifacts remain.
The layered take: word boxes fixed geometry, the rejoin fixed numeric formatting, the
clean join fixed text formatting and the persisted data - and none whitewashed a real
error. The single-token guard from the finding below still holds throughout.

### Finding - two dominant content failure types (not "OCR is bad")

Per-cell GT-vs-OCR diff on topology-matched samples (`scripts/diff_content.py`) showed
content error analysis with two dominant failure types:

1. **Financial formatting normalization gaps**, especially dot leaders ("Label . . . .
   45,854") and currency markers (OCR reading `$` as `S`, or a stray trailing letter).
   Here OCR was actually correct; the eval was too strict. Fixed in
   `normalize_financial_number` (strip leader dots, extract the single numeric token,
   tolerate stray markers) - commit `05563c8`.
2. **Spatial column grouping errors** where adjacent numeric columns are merged into one
   cell ("2011 2010", "$10,376 $ 9,812") or shifted one column right (leaving the left
   cell empty). These are real extraction errors and are deliberately NOT fixed by
   normalization: the single-numeric-token rule returns None for multi-number cells, so
   they stay failures. Next step for this class is spatial/overlay debugging (column
   boundaries / word grouping), not text normalization.

Key point: low cell_text_exact_match / numeric_cell_relaxed_match is NOT "PaddleOCR is
20% accurate". On the topology-matched subset (perfect grid, mean IoU 1.0) the loss was a
mix of (1) eval strictness and (2) column grouping - the OCR text itself reads correctly.

## 2026-05-30 - Phase 1A-colab TATR bring-up

Context: first time running the real TATR structure model on Colab T4 via
`scripts/run_phase1a_colab.py` (notebook `01_phase1a_tatr.ipynb`). The CPU logic
(manifest, topology metrics, failure logger, XML parser) was already green under local
`pytest`; these failures are all in the GPU-only inference path, which cannot be tested
locally on Windows, so we iterated through the Colab paste-back loop.

How we read the evidence each round: the runner never aborts on a bad sample - it logs
to `outputs/failure_logs/phase1a_<run>.jsonl` and records `failed` in the manifest. So
each failed run gave us one precise error string to fix. The `failures.jsonl` is
append-only, so old runs stay in the file; we read the entries by their newest
timestamp (or use a fresh `--run-id` for a clean log).

### Timeline and effort (UTC)

| Time (UTC) | Event |
|-----------|-------|
| 13:51 | `bce6e20` runner first pushed (initial Phase 1A-colab) |
| 13:56 | Colab run 1 -> SizeDict error (Issue 2) |
| 14:02 | `fd5e598` notebook rebuilt (Issue 1) |
| 14:12 | Colab run 2 -> still SizeDict (old code, pull was skipped) |
| 14:16 | `6b71082` use_fast=False (Issue 2 fix) |
| 14:19 | Colab run 3 -> size-key error (Issue 2 gone, Issue 3 surfaced) |
| 14:29 | Colab run 4 -> still size-key (before Issue 3 fix) |
| 14:34 | `1e6c7e5` shortest_edge added (Issue 3 fix) |
| after 14:34 | verification run (`--run-id smoke`, limit 10) -> **processed=10 failed=0** |
| 14:41 | green run's geometry flags read: `adjacent rows overlap` (real) + `row boxes not sorted` (false positive) |
| 15:04 | `4dab67a` run summary log + sort-before-validate fix |
| 15:07 | resume re-run (same `--run-id smoke`) -> `skipped=10`, report clobbered to zeros |
| 15:11 | `46b4dbc` guard: do not write the report when no samples were processed |
| 15:16 | `bd242c6` `evaluate_tables.py`: authoritative report recomputed from predictions |
| 16:00 | `mvp` run, limit 50 -> processed=50 failed=0; row 0.84 / col 1.0 / occ 0.982 / span 1.0 |
| 16:04 | `mvp` resumed to limit 300 -> processed=250 skipped=50 failed=0; topology on a **non-random first-300 (ADS-dominated) subset** (see below) |
| post-`77e422d` | `mvp_rand` (seed 42, limit 300) -> row 0.79 / col 0.987 / occ 0.977 / span 0.957 - the representative Phase 1A number |

- **Active debugging span:** ~43 min (13:51 -> 14:34 UTC), from first runner push to the
  third fix.
- **Iterations:** 3 fix commits, 5 Colab inference runs (the 5th was green).
- **Outcome:** Phase 1A-colab TATR path working end to end on a 10-sample smoke batch.
- **Token usage:** not tracked here - I cannot read this session's token count, so I will
  not invent one; check the Claude usage dashboard for the real number. The iteration and
  run counts above are the honest effort proxy.

Commit times from `git log` are author-local (+0100); converted to UTC here so they line
up with the failure-log timestamps (which are UTC).

### Finding - display cell missed config.FIGURES (stale module cache, not a missing pull)

- **Symptom:** after `git pull`, the Step 3 *render* subprocess produced all figures fine,
  but the in-notebook *display* cell raised `AttributeError: module 'src.config' has no
  attribute 'FIGURES'`.
- **Root cause:** `!python scripts/render_phase1a_figures.py` runs in a fresh subprocess,
  so it reads the pulled `config.py` (with `FIGURES`). The display cell runs in the kernel,
  which imported `src.config` back at the boot cell - before the pull - and Python caches
  modules, so the in-memory `config` never gained `FIGURES`.
- **Fix:** `importlib.reload(config)` at the top of the display cell, so it picks up
  config changes pulled after the kernel started (no runtime restart needed). `182ef66`.
- **Lesson:** in Colab, a `!python` step always sees fresh code; in-kernel `from src
  import X` does not. After pulling code that changes a module the kernel already imported,
  reload it (or restart the runtime).

### Result - Phase 1A topology FINAL (representative random subset, seed 42)

- **The number to report.** `mvp_rand`, a fixed random 300-table subset (seed 42, drawn
  across all issuers/years), metrics over successful samples; processed=300, failed=0:
  - row_count_accuracy = 0.79
  - col_count_accuracy = 0.987
  - cell_occupancy_f1 = 0.977
  - spanning_cell_detection_rate = 0.957
- **Why it differs from the ADS subset (0.897 row):** the alphabetically-first 300 were
  almost all one issuer (ADS), whose table styles TATR handles unusually well on rows.
  The random sample exposes more row over-segmentation, so row accuracy drops to 0.79 -
  this is the honest, dataset-representative figure; the ADS number is superseded.
- **cell_occupancy_f1 = 0.977** stays high, which is the metric that matters most for
  table-only RAG (Phase 1C): cell positions are right even when the row count is off.

**Formal conclusion (report wording):**

> The earlier ADS-only 300-table run overestimated topology performance because the
> subset was alphabetically biased toward one issuer. On the fixed random 300-table MVP
> subset (seed=42), row-count accuracy drops from 0.897 to 0.790, showing that row
> segmentation is the main cross-issuer weakness. Column count remains near-perfect at
> 0.987, cell occupancy F1 remains strong at 0.977, and spanning-cell detection is stable
> at 0.957.

Always report alongside the metrics:
- Metrics are computed over successful samples only.
- Subset: fixed random 300-table MVP subset, seed=42.
- Run accounting: processed=300, skipped=0, failed=0.

### Result - Phase 1A topology on a non-random 300-table subset + decoupled-eval payoff

- **SAMPLING CAVEAT (important):** these 300 are NOT a random sample. `find_xml_files`
  does `sorted(rglob("*.xml"))[:limit]`, so the "first 300" are alphabetically-first
  filenames - in practice almost entirely one issuer's filings (`ADS_2007`/`ADS_2008`,
  i.e. ADS). So the numbers below describe that subset's table styles, NOT FinTabNet.c
  as a whole. Do not report them as a dataset-level result. A representative number needs
  a random (ideally stratified-by-issuer) sample; tracked as a follow-up. Always report
  alongside: processed / skipped / failed, and "metrics computed over successful samples."

- **What we ran:** `--run-id mvp` first at `--limit 50` (16:00), then resumed at
  `--limit 300` (16:04). The resume worked exactly as designed: `processed=250
  skipped=50 failed=0`. `find_xml_files` sorts then slices, so the first 50 are a
  subset of the first 300 - the 50-sample run's predictions were reused, not redone.
- **The decoupled-eval payoff, seen live:** the GPU runner's run-summary reported
  `num_samples=250` (only what *that* run processed), but `evaluate_tables.py` reported
  `num_samples=300` (every completed sample). The 300 figure is the authoritative one.
  Had evaluation stayed coupled to the run, the headline number would have silently
  covered only 250 samples. This is the concrete reason the two were split.
- **Convergence trend (row_count_accuracy):** 0.70 (10) -> 0.84 (50) -> **0.897 (300)**.
  The 10-sample 0.70 was small-sample noise; the metric settles near 0.90. The residual
  ~10% is TATR row over-segmentation (`adjacent rows overlap > 0.3`), a known TATR
  weakness, not a pipeline bug.
- **spanning_cell_detection_rate** moved 1.0 -> 1.0 -> **0.961**: the early 1.0 was
  small-sample optimism; harder spanning cases appear as N grows. 0.96 is the real rate.
- **Topology on the 300-table subset (authoritative `evaluate_tables.py`), metrics over
  successful samples; processed=300, skipped=0 (cumulative), failed=0:**
  - row_count_accuracy = 0.897
  - col_count_accuracy = 0.997
  - cell_occupancy_f1 = 0.988  (the key metric for table-only RAG)
  - spanning_cell_detection_rate = 0.961
  - 300 samples inferred in ~34 s on T4. See the sampling caveat above: subset, not
    dataset-level.
- **Lesson reinforced:** report the metric over all persisted artifacts, never over
  "what this run happened to touch." The run-summary's count is a run log, not a result.

### Finding - resume clobbered the report; decoupled evaluation (evaluate_tables.py)

- **What we saw:** re-running with the same `--run-id smoke` gave `processed=0
  skipped=10 failed=0` - resume correctly skipped the 10 already-`success` samples. But
  the runner recomputed the report from "what this run processed" (now empty), so it
  **overwrote a good report (0.7 / 1.0 / 0.966 / 1.0) with zeros**.
- **Root cause:** evaluation was coupled to the GPU run and scoped to one run's
  processed samples. A full skip -> empty -> zeros; a partial resume would also report
  only the newly processed subset.
- **Fixes (two layers):**
  1. `46b4dbc` - guard: only write the report when the run processed samples, so a
     fully-skipped resume no longer clobbers it.
  2. `bd242c6` - `scripts/evaluate_tables.py`: a CPU-only step that recomputes the
     report from **all** persisted predictions, via `run_manifest.read_completed()`
     (latest-status-wins success rows from the manifest). Correct regardless of how many
     resume sessions produced the predictions; no GPU. This also means a clobbered
     report can be restored without re-running inference.
- **Lesson:** keep evaluation separate from the (expensive, resumable) extraction run;
  the run produces artifacts, a separate pass scores them.
- **Commits:** `46b4dbc` (15:11 UTC), `bd242c6` (15:16 UTC)

### Finding - grid_geometry flags: real signal vs false positive; added run log

- **What we saw:** after the green run, the newest failure-log entries (14:41) were
  `error_type: grid_geometry`, NOT failures - `validate_grid_geometry` logs quality flags
  but does not fail the sample, so the run was still `failed=0`. Two messages:
  - `adjacent rows overlap > 0.3` - **real**. TATR predicted overlapping row boxes
    (row over-segmentation). This is the likely cause of `row_count_accuracy=0.7`
    (sample `ADS_2007_page_149_table_0` flagged repeatedly).
  - `row boxes not sorted top-to-bottom` - **false positive in our usage**. The runner
    passed the raw (unsorted) TATR boxes to the validator, but the grid is built from
    boxes sorted inside `normalize_tatr_prediction`, so order is not a real defect; it
    was just polluting the log.
- **Fix:** sort row/col boxes before `validate_grid_geometry`, removing the false
  positive while keeping the real overlap flags.
- **Also added:** a one-line run summary appended to `manifests/phase1a_runs.jsonl` each
  run (run_id, time, params, processed/skipped/failed, metrics). The presence of a line
  means the run completed - an explicit "this execution finished" record to complement
  the per-sample manifest and the failure log. The notebook inspect cell prints it.
- **Commit:** `4dab67a` (15:04 UTC)

### Issue 3 - TATR processor: missing size key (resolved)

- **Symptom:** all 10 samples fail with `error_type: tatr_inference`.
- **Error:** `Size must contain 'height' and 'width' keys or 'shortest_edge' and
  'longest_edge' keys. Got dict_keys(['longest_edge']).`
- **Root cause:** the `microsoft/table-transformer-structure-recognition-v1.1-fin`
  checkpoint ships `preprocessor_config.json` with `size = {'longest_edge': N}` only
  (it reflects the original TATR MaxResize semantics). The HF DETR image processor's
  resize step requires either `shortest_edge`+`longest_edge` or `height`+`width`, so it
  rejects a longest-edge-only size on every image.
- **Fix:** after loading the processor, add a `shortest_edge` while preserving the
  checkpoint's `longest_edge`:
  ```python
  longest = processor.size.get("longest_edge", 1000)
  processor.size = {"shortest_edge": min(800, longest), "longest_edge": longest}
  ```
  For wide table crops the binding constraint is `longest_edge`, so this approximates
  the original MaxResize. `target_sizes` in post-processing stays the original image
  size, so predicted boxes come back in original-crop coordinates (matching the GT XML
  boxes).
- **Commit:** `1e6c7e5` (14:34 UTC)
- **Result:** the next run (`--run-id smoke`, limit 10) gave `processed=10 skipped=0
  failed=0`. Topology metrics on the 10-sample batch: `row_count_accuracy=0.7`,
  `col_count_accuracy=1.0`, `cell_occupancy_f1=0.966`,
  `spanning_cell_detection_rate=1.0`. This validated the whole GPU path end to end:
  the model's `id2label` strings (`table row`/`table column`/`table spanning cell`)
  are correct (non-zero row/col counts, spanning detected), and the prediction ->
  canonical -> metrics chain works. `row_count_accuracy=0.7` (3/10 wrong row counts)
  is the known TATR row over/under-segmentation - the first thing to look at when
  scaling up.

### Issue 2 - TATR processor: SizeDict crash from the fast processor

- **Symptom:** all 10 samples fail with `error_type: tatr_inference`.
- **Error:** `'SizeDict' object has no attribute 'keys'`. The run also printed a
  warning: *"The image processor of type DetrImageProcessor is now loaded as a fast
  processor by default ... To continue using the slow processor, instantiate this class
  with use_fast=False."*
- **Root cause:** the new fast DETR image processor (`use_fast=True`, now the default)
  has a bug in post-processing on this transformers version - it calls `.keys()` on a
  `SizeDict`.
- **Fix:** force the stable slow processor:
  ```python
  processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=False)
  ```
  This changed the error (SizeDict gone), which surfaced Issue 3 - confirming the fix
  was effective.
- **Commit:** `6b71082` (14:16 UTC)

### Issue 1 - Notebook cells mangled by NotebookEdit

- **Symptom:** after inserting the Step 2 cells, `01_phase1a_tatr.ipynb` had the
  pip-install and the runner command **merged into one cell, placed before `git pull`**,
  and the Step 2 markdown was lost.
- **Root cause:** the `cell-N` ids shown by the Read tool are display labels, not the
  notebook's real cell ids. Passing them to `NotebookEdit` as `cell_id` targeted
  nothing, so inserts landed in the wrong place and chained incorrectly.
- **Fix:** rebuild the whole `cells` list with a script (preserving notebook metadata),
  giving a clean order: Boot -> Step 1 inspect -> Step 2 install GPU stack -> run TATR.
- **Commit:** `fd5e598` (14:02 UTC)

### Setup notes that caused confusion (not bugs)

- **`timm` was missing** from `requirements-colab.txt`; `TableTransformerForObjectDetection`
  needs it for the backbone. Added in `bce6e20`.
- **No `outputs/` folder in the repo tree.** This is by design: in Colab,
  `config.OUTPUT_ROOT` points to Drive (`/content/drive/MyDrive/FinDocStructRAG/outputs/`),
  and `.gitignore` excludes `outputs/` (big artifacts live on Drive, never in git, per
  PLAN P2). Look under the Drive `FinDocStructRAG/outputs/`, not the git repo.
- **Confirmed from the manifest:** structure XMLs live under
  `FinTabNet.c-Structure/test/*.xml` and crops under `images/`. Because the failures
  were `tatr_inference` (after image load), the image-by-filename lookup is working.

### Workflow reminders (so the loop stays fast)

- `src/` is the source of truth; Colab only `git pull`s and runs. Every fix is: edit
  locally -> push -> `git pull` on Colab -> re-run (PLAN P2). A run that still shows an
  old error usually means the pull was skipped.
- The runner is resumable: it skips `sample_id`s already marked `success`. Failed
  samples are retried on the next run.
- Paste back **text** (the `processed/skipped/failed` line, summary JSON, newest
  failure entries), not screenshots.
