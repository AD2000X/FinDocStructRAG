# Phase 3 — FUNSD relation branch (V1 baseline)

> Implementation brief for Phase 3. Committed in the repo (travels with `git pull` to Colab)
> so the references to it in `DEVLOG.md` and the `src/funsd_extraction.py` /
> `src/eval_funsd.py` module docstrings resolve. Status: V1 implemented and scored on
> `feature/phase3-funsd-relations`; headline held-out `test_50.qa_links` F1 0.727.

## Context

Phases 0-2 are merged to `main` (FinTabNet.c table topology + OCR content + table-only RAG +
DocLayNet layout crop). Phase 3 is a **FUNSD relation-linking baseline** over GT entities.

It is a deliberately standalone branch — it does **not** touch the RAG pipeline and only
wires into the demo in Phase 4. It is **annotation-only and CPU-only**: the FUNSD
annotation JSON already carries each entity's text, bbox, label, and GT `linking` pairs,
so scoring, the spatial heuristic, and the gold links never load image pixels. No GPU, no
Colab — this is the local "logic loop", fast `pytest`.

The task: given GT entities `{question, answer, header, other}`, predict which entities are
linked, scored P/R/F1 against GT `linking`.

## Locked design decisions

- **Predictor:** deterministic spatial heuristic, **per-answer argmax + distance gate**.
  Each `answer` picks its single highest-scoring `question`; emit the link only if the best
  score clears the gate. Distances normalized by the form's **median entity height** so one
  gate works across differently-scaled scans. (Rejected: per-question argmax -> under-predicts
  multi-answer questions; global threshold -> precision-fragile, needs a tuned cutoff.)
- **GT links:** dedupe `linking` to **undirected** `frozenset({id_a, id_b})` per form (FUNSD
  records links bidirectionally/duplicated). Then derive scopes:
  - `qa_links` (**primary**): pairs whose endpoints are one `question` + one `answer`,
    canonicalized to directed `(question_id, answer_id)`.
  - `all_links` (**secondary**): the full deduped undirected set, scored as frozensets.
- **Eval split / reporting matrix:**
  - **Primary headline:** `test_50.qa_links.micro_f1` (official 50-form test split).
  - Set/tune any heuristic params on **train_149 only**; never on the reported test set.
  - **Secondary:** `all_199.qa_links`, `test_50.all_links`, `all_199.all_links`. Print
    `all_199` with a "contains the 50 test + 149 tuned forms, not held-out" caveat.
  - `debug_20` (first 20 train forms) is for parser/CLI smoke only, never for tuning.
- **all_links is a coverage diagnostic, not a second predictor.** V1 predicts only `q->a`;
  the `all_links` row scores those same QA predictions (as undirected frozensets) against the
  full GT link set, i.e. "what fraction of all GT links does the QA-only heuristic cover."
- **No sklearn in V1.** P/R/F1 is set arithmetic (`len(pred & gold)/len(pred)` etc.). sklearn
  enters only if a fitted ranker is added later, and only fit on train_149.
- **Data:** raw FUNSD zip -> `data/raw/funsd/...` (gitignored). Tests use **synthetic fixtures
  only**, never the raw dataset.

## Files

### `src/config.py` — FUNSD paths
```python
FUNSD_ROOT  = DATA_ROOT / "raw" / "funsd" / "dataset"
FUNSD_TRAIN = FUNSD_ROOT / "training_data" / "annotations"   # 149 forms
FUNSD_TEST  = FUNSD_ROOT / "testing_data" / "annotations"    # 50 forms
```
Output reuses existing `config.EVALUATION`.

### `src/funsd_extraction.py` — data contract + baseline predictor
Follows the `from __future__ import annotations` + TypedDict + pure-function style of
`src/canonical_schema.py` / `src/eval_retrieval.py`.
- `FunsdEntity` TypedDict: `id, label, text, box [x0,y0,x1,y1]`.
- `FunsdForm` TypedDict: `form_id, entities, gold_links (set[frozenset[int]])`. In-memory it
  is a `set` (dedupe is its nature); cast to `list` only when serializing JSON output.
- `parse_funsd_form(data, form_id)` / `parse_funsd_json(path)`: normalize entities, collect
  `linking`, dedupe to undirected frozensets, drop self-links and links to missing ids.
- `load_funsd_split(annotations_dir)`.
- `qa_gold_links(form) -> set[tuple[int,int]]`: question+answer pairs -> directed `(q,a)`.
- `all_gold_links(form) -> set[frozenset]`: the deduped undirected set.
- `HeuristicParams` (frozen dataclass, a-priori defaults; the documented tunable surface,
  train_149 only). Two clearly-separated knobs, not one fuzzy "gate":
  - `max_distance_units`: **distance gate** — a (Q, A) candidate is rejected if its
    median-height-normalized distance exceeds this. Filters the candidate set.
  - `min_score`: **score threshold** — the per-answer argmax winner is emitted only if its
    final score clears this. Acceptance test on the chosen link.
  - plus right-band tolerance, below-gap tolerance, and the boost weights.
- `predict_qa_links(form, params=HeuristicParams()) -> set[tuple[int,int]]`: per-answer argmax.
  For each answer A, score **every** question candidate (distances scaled by
  `median_entity_height(entities)`), drop candidates beyond `max_distance_units`, take the
  highest-scoring question, and emit `(q,a)` only if that score >= `min_score`. Geometry:
  - **same-row right-side** (A vertically within Q's band, A to Q's right): strongest score.
  - **below** (A under Q, horizontally aligned/overlapping): fallback score.
  - proximity + alignment are additive boosts. (below and right-side compete in the same
    argmax — a below candidate wins whenever it is the best valid candidate.)

### `src/eval_funsd.py` — custom set-based metrics
Split so the metric stays pure (no predictor/params dependency) and the form-runner is separate:
- `prf1(pred, gold) -> dict`: one pred/gold set -> tp/precision/recall/f1, zero-guards.
- `evaluate_pairs(per_form) -> dict`: **pure** micro P/R/F1 over prebuilt (pred, gold) pairs +
  counts. No predictor inside, so it is trivially unit-tested with synthetic sets.
- `evaluate_forms(forms, scope, params=HeuristicParams()) -> dict`: builds per-form (pred, gold)
  for `scope in {"qa","all"}` — qa = directed `(q,a)` tuples from `predict_qa_links` vs
  `qa_gold_links`; all = those predictions cast to undirected frozensets vs `all_gold_links` —
  then delegates to `evaluate_pairs`.

### `scripts/evaluate_funsd.py` — CLI runner
Mirrors `scripts/evaluate_rag.py`. Loads `train_149` / `test_50`, builds `all_199` and
`debug_20`, runs split x scope, writes `config.EVALUATION / "phase3_funsd_relations.json"`,
prints the headline `test_50.qa_links` + secondaries. Guards with a `SystemExit` pointing at
`fetch_funsd.py` when the dataset is missing.

### `scripts/fetch_funsd.py` — one-time data helper
`urllib` + `zipfile` download/extract of the official FUNSD zip to `data/raw/funsd/`, with a
`--url` override and a printed manual-download fallback. Not used by tests; not on the gate.

### `tests/test_funsd_relations.py` — synthetic fixtures only (acceptance gate)
Inline tiny forms (dicts), no raw dataset. Covers: parse + gold_links; bidirectional/duplicate
dedupe; qa-link filter and direction canonicalization; all_links scope; `prf1` / `evaluate_pairs`
edge cases; same-row right-side link; below-candidate-wins-when-best; `other` never linked;
per-answer argmax picks the nearer; distance gate; header->question excluded from QA; and two
form-level `evaluate_forms` cases.

### `notebooks/05_phase3_funsd_relations.ipynb` — Colab/local runner (no logic)
Mount/pull/fetch/test/evaluate/display + a read-only qualitative error table.

## Out of scope (V1)
- FUNSD token classification (V2 / seqeval) — future work.
- Image/overlay loading — optional later debug aid, not in the baseline or the gate.
- Any RAG integration — Phase 4.
- sklearn / fitted rankers.

## Result (real FUNSD, untuned a-priori params)

Headline (held-out): `test_50.qa_links` P 0.946 / R 0.590 / **F1 0.727**. `train_149` F1 (0.665)
is below test, so there is no tuning-on-test. Recall is the design ceiling (single-link per
answer + right-side/below geometry); threshold-based multi-link is the documented next lever.
Full split x scope matrix in `DEVLOG.md` (2026-06-03) and
`outputs/evaluation/phase3_funsd_relations.json`.

## Verification
1. **Unit (the gate):** `pytest tests/test_funsd_relations.py` green — fully synthetic, local,
   no GPU/network. Full suite 236 passed.
2. **End-to-end (needs the dataset, still local/CPU):** `python scripts/fetch_funsd.py` then
   `python scripts/evaluate_funsd.py` -> writes `outputs/evaluation/phase3_funsd_relations.json`.

## Branch / workflow
- This brief is committed at `docs/phase3_brief.md`. The `plans/` directory stays gitignored
  for local scratch (harness plan file, PR body draft); the canonical brief lives here.
- Branch `feature/phase3-funsd-relations` cut from `origin/main`. Entirely local phase — no
  Colab round-trip needed.
- Build order (TDD): fixtures+tests -> `funsd_extraction.py` -> `eval_funsd.py` ->
  `evaluate_funsd.py` -> `fetch_funsd.py` -> docs.
