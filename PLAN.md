# FinDocStructRAG — Execution Plan

> Layout-aware financial document intelligence pipeline: PDF layout parsing,
> financial table extraction, OCR/form relation linking, and retrieval-grounded QA
> under Colab T4 constraints.
>
> This document is the **execution roadmap** derived from the V9 design spec
> ([DESIGN_SPEC.md](DESIGN_SPEC.md)). The V9 spec defines
> *what* to build (schemas, function signatures, metrics, tests). This `PLAN.md`
> defines *the order, environment workflow, and acceptance criteria* for building it.

---

## 0. Tech Stack Decisions

The choices below are **locked**; implementation does not re-litigate them without a
strong reason. Items marked "experiment" are decided by Phase results; items marked
"optional" are non-essential.

| Area | Decision | Source |
|------|----------|--------|
| Datasets | FinTabNet.c / DocLayNet / FUNSD / self-built QA | V9 |
| Table structure recognition | `microsoft/table-transformer-structure-recognition-v1.1-fin` | V9 §4.2 |
| Table detection (fallback) | `microsoft/table-transformer-detection` | V9 §4.2 |
| OCR | PaddleOCR (priority) > Tesseract | V9 §5.5/5.11 |
| Retrieval | BM25 + dense (exact BGE cosine; FAISS optional if the corpus grows) + RRF + type-aware reranking + rule-based routing | V9 §7 |
| Reranking | heuristic / rule-based (cross-encoder is future work) | V9 §7 |
| **Embedding model (dense retrieval)** | **`BAAI/bge-small-en-v1.5`** | This plan (V9 unspecified) |
| **Answer-generation LLM** | **OpenRouter (OpenAI-compatible gateway; default `openai/gpt-4o-mini`; single provider, swappable via `src/llm_client.py`)** | This plan |
| Numeric normalization | `src/numeric_utils.py` (V9-fixed `looks_numeric`) | V9 §5.13 |
| Eval tooling | pycocotools / custom retrieval metrics (hit/recall/MRR; ranx optional) / seqeval / sklearn / GriTS (final) / custom proxy | V9 §8 |
| Demo | Gradio | V9 |
| Environment / workflow | VS Code Colab extension + git-as-truth + `.py` core / `.ipynb` runner | This plan §1 |

**Deliberately undecided (not an omission)**
- Table serialization: Markdown vs linearized -> **resolved in Phase 1C: linearized carries forward** (reproducible +0.10 numeric_relaxed over markdown on both the gt and ocr corpora, temperature=0)
- RAG faithfulness (Ragas/DeepEval), LLM-as-Judge -> **optional** (V9 §7/§8)
- HyDE / LLM rewriting / cross-encoder -> **future work** (V9 §7)
- **Page-level layout detector model** (the model that produces DocLayNet-class regions) -> **pin before Phase 2 coding** (§3 Phase 2). DocLayNet is the dataset / AP-IoU target, not a model; the spec (§9) assumes a HF detector exposing `config.id2label`. Candidates and recommendation are in the Phase 2 section.

---

## 1. Environment: local VS Code + Google Colab extension

Connection: the **official Google Colab extension for VS Code** (edit notebooks locally,
execute on a remote Colab kernel).

### Key constraint (verified)

> **The Colab kernel cannot see local files.** A `.ipynb` lives locally and opens in VS Code,
> but once a Colab kernel is selected, the kernel's filesystem is the Colab VM's `/content`.
> `pd.read_csv("local_file.csv")` fails, and the local `src/` is invisible to the kernel.
> ([googlecolab/colab-vscode#210](https://github.com/googlecolab/colab-vscode/issues/210))

### Three storage locations (independent of each other)

| Location | Path | Characteristics |
|----------|------|-----------------|
| (1) Local disk | `...\FinDocStructRAG` | VS Code editing surface, git working copy |
| (2) Colab VM scratch disk | `/content/...` | Colab host disk, **wiped when the session ends (ephemeral)** |
| (3) Google Drive | `/content/drive/MyDrive/...` | Cloud persistent storage, mounted into the Colab VM |

### Where each kind of file lives

| File | Location | Sync method |
|------|----------|-------------|
| `.ipynb` notebooks | **Local** (VS Code workspace) | Edited directly; at run time the kernel is on the Colab VM |
| `src/` `tests/` code | Edit locally -> push to GitHub -> Colab notebook `git pull` onto the VM | Note: the kernel cannot see the local `src/`; it **must be cloned to the VM to import** |
| `data/`, model weights, `outputs/` | **Google Drive** | Mounted into the VM; not in git; survives session loss |

**Core mental model**: you write code locally, but the Colab kernel runs a **separate copy**
it pulled from GitHub into the Colab VM's `/content`. The two copies stay in sync through
**git as the single source of truth**.

### Dual dev loops

- **Local loop (CPU, seconds)**: pure-Python logic (`tatr_postprocess`, `numeric_utils`, `html_to_canonical`, `bbox_utils`, `chunking`) plus all of `tests/` -> local `pytest`, never touching Colab.
- **Colab loop (GPU)**: GPU-bound inference (TATR, PaddleOCR, DocLayNet) and dense-embedding / index building -> notebook runs on the Colab kernel.

### Colab notebook role: runner, not source code

Every Colab notebook starts with the same five steps and **does not hold large function
definitions** (all functions live in `src/`):

```python
# 1. Mount Drive (persist data/outputs)
from google.colab import drive; drive.mount('/content/drive')

# 2. git pull the latest code onto the VM (so the kernel can see src/)
!cd /content/FinDocStructRAG 2>/dev/null && git pull || git clone <repo-url> /content/FinDocStructRAG

# 3. Install Colab-only dependencies
!pip install -q -r /content/FinDocStructRAG/requirements-colab.txt

# 4. Make src/ importable
import sys; sys.path.insert(0, '/content/FinDocStructRAG')
from src import config, tatr_postprocess   # ...

# 5. Run -> outputs always written to Drive (config detects the environment and returns a Drive path)
```

**Anti-pattern to avoid**: editing `src/` directly in Colab -> it diverges from
local/GitHub. The Colab side only "runs" and "produces outputs".

### Formal rule: `.py` is the core, `.ipynb` is a runner

Core logic is always written as `.py`; notebooks are an execution and presentation layer
and do not accumulate logic.

| Type | Purpose | Location |
|------|---------|----------|
| `.py` | Core logic, testable modules (function/class/schema/parser/metric/retrieval/eval) | `src/` |
| `.py` | Repeatable pipeline runners | `scripts/` |
| `.py` | unit / smoke tests | `tests/` |
| `.ipynb` | Colab GPU runner, result inspection, chart display | `notebooks/` |
| `.ipynb` | demo / report-style walkthrough | `notebooks/08_demo.ipynb`, `09_final_report.ipynb` |

**Decision rule**: function, class, schema, metric, parser, post-processing,
retrieval/eval logic, unit test, repeatable script -> always `.py` (e.g.
`html_to_canonical()`, `boxes_to_grid()`, `map_spanning_bbox_to_grid()`,
`assign_words_to_cells()`, `normalize_financial_number()`, RRF fusion). Notebooks only hold:
mount Drive, git pull, install, load sample, call a script/function, show images/tables,
debug, demo walkthrough.

**Key mechanism: the Colab extension gives you a kernel, not a VM terminal.** A GPU-bound
script cannot run in the local terminal (no local GPU); trigger it from a notebook cell with
`!python` so it runs on the Colab VM. The GPU flow therefore stays a repeatable,
version-controlled `.py`, and the notebook cell `!python ...` is the entire content.

| script | How to trigger | Runs on |
|--------|----------------|---------|
| `run_phase1a_local.py`, `evaluate_*.py` (pure CPU) | local terminal `python scripts/...` | Local |
| `run_phase1a_colab.py`, OCR/layout (need GPU) | notebook cell `!python scripts/...` | Colab VM |

Main dev loop: write `src/*.py` + `tests/*.py` locally -> local `pytest` passes ->
`git push` -> Colab notebook `git pull` -> notebook calls `src/` or `scripts/` -> outputs
written to Google Drive.

---

## 2. Phase order

Principle: **build the smallest demonstrable closed loop first**, strictly in sequence, no
parallel expansion. RAG appears for the first time in Phase 1C (matching the project name
FinDocStructRAG) rather than waiting until the end.

```
Phase 0  Repo foundation              ~0.5 day   all local
Phase 1A Table topology               3-4 days   mostly local + a little Colab for TATR
Phase 1B OCR content extraction       2-3 days   mostly Colab (PaddleOCR)
Phase 1C Table-only RAG QA            2-3 days   mixed (index on Colab, logic local)
Phase 2  DocLayNet layout integration 1-1.5 wks  mostly Colab
Phase 3  FUNSD relation branch        1 week     mostly local (GT tokens, no GPU)
Phase 4  Full demo + evaluation       2-3 days   mixed
```

**Timeline note**: the day counts above are "frictionless ideal time" and do not account
for Colab session drops, FinTabNet.c annotation-format surprises, PaddleOCR install pitfalls,
and other real-world resistance. **Realistically about 3-5 weeks of calendar time**; the
biggest schedule risk is these environment/data surprises, not the coding itself.

**Release split**: **v1 release = Phase 1C** (table-only RAG end-to-end demo), the first
demonstrable version. Phases 2-4 are **follow-on releases** — still required Phases to
complete V9 (not optional), they just do not block the first demonstrable build.

---

## 3. Per-phase deliverables and acceptance criteria

### Phase 0 — Repo foundation (first delivery)

**Deliverables**
- Complete V9 folder structure (`src/`, `tests/`, `notebooks/00-09`, `data/`, `outputs/...`, `assets/`, `reports/`)
- `src/` **locks only the interfaces Phase 1A actually depends on**; do not stub every future module at once (that is form-work that rots). What Phase 0 should fix: `config.py`, the canonical schema, `failure_logger.py`, the `tatr_postprocess.py` interface, the tests structure. Other modules are built when their own Phase starts.
- **Requirements split into three files + one entry point**:
  - `requirements.txt` — local install entry point (matches repo convention, the first thing GitHub users look for). Content:
    ```
    -r requirements-core.txt
    -r requirements-dev.txt
    ```
    README says `pip install -r requirements.txt`; the Colab notebook uses `pip install -r requirements-colab.txt`.
  - `requirements-core.txt` — local development: `pandas`, `bs4`, `numpy`, `lxml`, the LLM SDK (`openai`, used for the OpenRouter gateway; pure API calls, no GPU dependency, runs locally too)
  - `requirements-colab.txt` — GPU: `torch`, `transformers`, `paddleocr`, `faiss`, `datasets`
  - `requirements-dev.txt` — tooling: `pytest`, `ruff`, `black`, `mypy` (optional)
- `.gitignore` (excludes `data/`, `outputs/`, `*.pt`, the Drive mount point)
- `scripts/` — repeatable pipeline runners (rerun a pipeline from the command line, locally or on Colab). Notebooks are for inspection and demo; scripts are for reproducible execution. Initial set:
  ```
  scripts/
  ├── run_phase1a_local.py     # synthetic logic validation
  ├── run_phase1a_colab.py     # TATR inference + topology metrics
  ├── build_table_chunks.py    # Phase 1C chunk building
  ├── evaluate_tables.py
  └── evaluate_rag.py
  ```
- `src/config.py` — centralizes paths and model IDs, and **detects Colab vs local** to switch the root path automatically. Fix the path variables so each notebook does not redefine them:
  ```python
  PROJECT_NAME    = "FinDocStructRAG"
  DRIVE_ROOT      = Path("/content/drive/MyDrive") / PROJECT_NAME   # Colab data/outputs persistence layer
  COLAB_REPO_ROOT = Path("/content/FinDocStructRAG")                # git clone on Colab
  # Locally ROOT = project directory; DATA_ROOT / OUTPUT_ROOT point to local or DRIVE_ROOT by environment
  ```
- Canonical table schema fixed with a dataclass / TypedDict (a cross-cutting dependency; fix it as early as possible)

**Phase 0 must not be all-skip**; have at least 4 smoke tests that actually run (local):

```python
test_import_src()             # from src import ... does not error
test_config_paths()           # config path detection (Colab/local) returns valid values
test_failure_logger_init()    # failure_logger can be initialized
test_canonical_schema_type()  # canonical schema types are correct
```

**Acceptance criteria**
> pytest runs with **at least 4 smoke tests passing**.
> Heavy tests are marked `skip`/`xfail`.
> `from src import ...` works locally and after `git pull` on the Colab VM.

---

### Phase 1A — Table topology (first delivery, core)

Split into two steps: validate pure logic locally with synthetic data first, then run real
TATR prediction on Colab to compute topology metrics. Rationale: `boxes_to_grid()` and the
like can be unit-tested with fake boxes, but real topology metrics require an actual TATR
prediction.

#### Phase 1A-local (local CPU, following the V9 dependency order, each step with a synthetic unit test)

1. `html_to_canonical()` (occupancy-aware, V9 gives a complete implementation) — GT parsing relies on it, so do it first
2. FinTabNet.c annotation gate (`can_convert_to_canonical`) — reject malformed formats up front
3. `boxes_to_grid()` + `validate_grid_geometry()`
4. `map_spanning_bbox_to_grid()` + `apply_spanning_cells()` (a V9 addition, a key focus)
5. unit tests with **synthetic boxes** (fake row/col/spanning boxes, no TATR needed)

**Phase 1A-local acceptance**
- All 5 spanning tests listed in V9 pass: `test_map_spanning_bbox_covers_two_rows`, `..._three_cols`, `..._no_overlap_returns_none`, `test_apply_spanning_cells_merges_correctly`, `..._removes_covered`
- Numeric tests such as `test_looks_numeric_parentheses_no_digits` ("Operating Income (Loss)" -> False) all pass

#### Phase 1A-colab (Colab T4, real TATR)

1. Run TATR structure recognition on 30-50 FinTabNet.c cropped table images
2. Save predicted row / column / header / spanning boxes
3. Feed predictions back into the Phase 1A-local logic (`boxes_to_grid` -> `apply_spanning_cells` -> canonical)
4. **Strictly separate `gt_filled/` and `tatr_predicted/` outputs** (metadata records `text_source` + `evaluation_type`; GT text is never treated as an extraction result)
5. Topology metrics: row/col count accuracy, `cell_occupancy_f1`, `spanning_cell_detection_rate`, header detection accuracy, parse/html success rate
6. 9 deliverable screenshots (see V9 §5.14) + `failure_logger.py`

**Phase 1A-colab acceptance**
- Produce the topology metrics table on the 30-50 image debug subset
- All 9 screenshots present

---

### Phase 1B — OCR content extraction

Mostly Colab. `PaddleOCR` (priority over Tesseract) -> `OCRWord` dataclass ->
`assign_words_to_cells()` (center-in-cell -> max IoU fallback -> unassigned logging) ->
`numeric_utils` (V9-fixed `looks_numeric`) -> content metrics.

**This is where "real extraction QA" is unlocked**: Phase 1A's QA is only pipeline
validation (carries a disclaimer).

**Acceptance criteria**
- `assign_words_to_cells()` tests pass
- content metrics (`cell_text_exact_match`, `numeric_cell_relaxed_match`, `non_empty_cell_content_f1`) produced on the MVP subset
- Push GitHub V1

---

### Phase 1C — Table-only RAG QA (RAG's first closed loop)

**Why here**: the project is named FinDocStructRAG, so RAG should not wait until Phase 4. As
soon as table extraction is done, attach table-only RAG to form the earliest complete demo
loop.

**Non-goal (strictly bounded)**: Phase 1C does **table-only RAG only**, with a corpus
limited to reconstructed table chunks. **Do not do full-document RAG in 1C** (a mixed
text/layout/form corpus waits until after Phase 2 layout integration). Otherwise scope creep
pushes Phase 2 back.

```
Phase 1C: Table-only RAG QA
- Use only extracted / reconstructed tables (Phase 1A/1B outputs)
- Build table chunks (metadata-preserving)
- Compare Markdown vs linearized table serialization
- BM25 + dense retrieval (exact BGE cosine, BAAI/bge-small-en-v1.5; FAISS optional if the corpus grows)
- RRF fusion
- Source-grounded QA (with citations)
- Report GT-filled QA vs OCR-filled QA separately
- Numeric cross-check using numeric_utils.py
```

**LLM is used only in answer generation**: retrieval has no LLM at any point (BM25 + dense
BGE cosine + RRF + type-aware reranking). Only the final step "generate an answer from evidence" uses a
**single API LLM**, through the `src/llm_client.py` abstraction (see §4 guideline).

**Acceptance criteria**
- End-to-end: query -> retrieve table chunk -> grounded answer + source citation
- Markdown vs linearized serialization, two comparison results
- GT-filled QA and OCR-filled QA metrics reported separately (never mixed)
- Retrieval metrics (custom hit@k / recall@k / MRR@k; `ranx` optional)

---

### Phase 2 — DocLayNet layout integration

**Scope / why now**: Phase 1 consumed *pre-cropped* FinTabNet.c table images. Phase 2 closes
that gap on the input side: detect regions on a **full page**, normalize their labels, and
crop the table regions so they enter the **existing, unchanged** Phase 1A topology + Phase 1B
OCR pipeline. The carry-forward is the table crop; nothing downstream of the crop changes.

**Non-goal (bounded, same discipline as 1C)**: Phase 2 is layout detection + table-crop
handoff only. It does **not** build full-document RAG over text/figure/caption regions (a
mixed-corpus retrieval task that waits until the layout output is trusted). Detecting
non-table classes is for AP/IoU scoring and the future corpus, not for 1C-style answering yet.

#### Decision to lock first: the page-level layout detector

DocLayNet is the **dataset and the AP/IoU target**, not a model. The pipeline needs a detector
*trained on* DocLayNet classes whose output feeds §9's `id2label -> LAYOUT_LABEL_MAP ->
normalize_label`. On a T4 this is a **pretrained detector at inference**, not training from
scratch. Candidates:

| Option | API | Notes |
|--------|-----|-------|
| **HF DETR-family fine-tuned on DocLayNet** (deformable-DETR / RT-DETR / DiT variant) | `transformers.AutoModelForObjectDetection` | **Recommended.** Exposes `config.id2label` exactly as §9 assumes; no new heavy dep (transformers already in the stack); T4-friendly. One weight download. |
| YOLO-DocLayNet (ultralytics) | ultralytics `YOLO` | Fastest on T4, but a non-transformers API (no `config.id2label`), adds the ultralytics dep, needs a label-map shim. |
| PP-DocLayout (PaddleX) | Paddle | Strong, but pulls the Paddle stack (a Phase 1B install pain point) and a non-HF API. |
| Detectron2 / LayoutParser | Detectron2 | **Avoid**: brittle install on Colab T4. |

Recommendation: the **HF DETR-on-DocLayNet** path, to keep §9 literal and add no heavy
dependency. The exact model id is pinned into `config.LAYOUT_MODEL` only after it is confirmed
to load and run on a T4 (verify before locking, like the embedding model was).

#### Architecture (DESIGN_SPEC §4.1, sequential-first + fallback)

```
page image -> layout detector -> regions {Text, Title, Table, Figure, ...}
  -> normalize_label -> select Table regions
  -> bbox_utils crop (with small padding)
  -> [low-confidence / empty: fallback to microsoft/table-transformer-detection]
  -> IoU dedup -> each table crop -> EXISTING Phase 1A TATR + Phase 1B OCR (unchanged)
```

#### Phase 2-local (CPU, pure logic, synthetic unit tests first — P3)

1. `src/bbox_utils.py` (§4.3, the first step): coordinate/format conversions (xyxy<->xywh,
   COCO<->pixel, DocLayNet normalized 0-1 <-> page pixels at a pinned render DPI),
   crop-with-padding, IoU, IoU-dedup / NMS-merge. All pure, fake-box unit tests, no GPU.
2. `src/layout_parsing.py`: `LAYOUT_LABEL_MAP` + `normalize_label()` (§9), a region dataclass,
   and the **sequential + fallback** selection/dedup as a pure function over detector outputs
   with the **detector injected** (the `llm_client.complete` pattern), so the whole
   page->crops path is testable with fake detections and no GPU.
3. `src/table_detection.py`: the fallback table-detector adapter
   (`microsoft/table-transformer-detection`, already in config) behind the same region contract.
4. Tests: `test_label_mapping.py` (§12/§13), bbox conversions / IoU / dedup, the
   fallback-trigger rule (low confidence -> fallback fires; high confidence -> it does not),
   crop-coordinate correctness.

#### Phase 2-colab (Colab T4, real detection)

1. DocLayNet access via HF `datasets`: `unique("doc_category")` first, then filter (§2); fixed
   random subsets per §18.6 (debug seed 7 / mvp seed 42).
2. Run the layout detector on a DocLayNet page subset; persist predicted regions + table crops
   to a **Phase-2-owned** artifact stream (`outputs/layout/`), kept separate from the FinTabNet
   table outputs (P4 separation discipline).
3. Feed detected table crops through the **existing** Phase 1A TATR topology pipeline to confirm
   the handoff end-to-end (full page -> detected crop -> grid), the FinTabNet->page generalization.
4. Layout AP/IoU via `pycocotools` against DocLayNet GT regions.

#### New config / artifacts

- `config.LAYOUT_MODEL` (id pinned after the T4 load check), DocLayNet paths, a pinned render DPI.
- `outputs/layout/` (predicted regions + detected crops) and `outputs/manifests/phase2_layout_<run>.csv`
  per the §18 manifest convention; detected crops never mixed into the FinTabNet streams.

**Acceptance criteria**
- `bbox_utils`, `normalize_label`, and the sequential+fallback selection unit tests pass locally (P3).
- Layout AP/IoU reported on a fixed DocLayNet subset (`pycocotools`), with `processed / skipped /
  failed` + the subset descriptor (§18.6) — never phrased as a whole-dataset number.
- End-to-end: a table crop detected from a full page runs through the existing Phase 1A TATR
  topology pipeline and produces a grid.
- P4 held: DocLayNet GT regions vs detected regions stay separate artifacts; detected crops are
  not reported as GT.

**Risks specific to Phase 2**
- Coordinate-space mismatch (DocLayNet normalized 0-1 / COCO xywh / PDF points / rendered-image
  pixels) -> centralize every conversion in `bbox_utils`, pin one render DPI.
- Double counting between the layout-table path and the table-transformer-detection fallback ->
  IoU dedup with a fixed, unit-tested threshold.
- A DocLayNet "Table" crop may be looser than what TATR-fin expects (TATR trained on tight
  crops) -> small padding, validate the handoff on the debug subset before scaling.
- Detectron2 / Paddle install brittleness -> the HF transformers detector avoids it.

---

### Phase 3 — FUNSD relation branch (V1 baseline)

Mostly local (GT tokens/entities, no GPU, can run in parallel with other Phases). spatial
heuristic + boosts -> relation P/R/F1 (custom + sklearn).

**Acceptance criteria**: `test_funsd_relations.py` passes; relation P/R/F1 on the 199 FUNSD
samples.

---

### Phase 4 — Full demo + evaluation + report

- Gradio demo (integrates layout -> table -> OCR -> RAG QA)
- Full evaluation: retrieval (`ranx`), RAG faithfulness (Ragas/DeepEval, optional), analysis of the 5 RAG error categories
- Final report + README Limitations (use the V9 §14 passage)
- GriTS (Final / stretch)

---

## 4. Cross-cutting concerns

- **Canonical schema consistency**: `normalize_table_annotation()`, `normalize_tatr_prediction()`, and `html_to_canonical()` must all emit the same schema, otherwise evaluation will not line up -> fix it with a dataclass/TypedDict in Phase 0.
- **Strictly separate topology vs content metrics** (V9 §6.3).
- **Subset size tiers**: Debug -> MVP -> Final; each Phase runs on the Debug subset first before scaling up.
- **Colab sessions drop easily**: all heavy-compute outputs land on Drive; notebooks/scripts are designed to be rerunnable and resumable (idempotent / resumable).
- **Output manifest (resume mechanism)**: each finished batch writes a manifest; the runner reads it at startup to skip already-completed samples and avoid reruns.
  ```
  outputs/manifests/
  ├── phase1a_tatr_predictions_manifest.csv
  ├── phase1b_ocr_manifest.csv
  └── phase1c_rag_manifest.csv
  ```
  Columns per row: `sample_id, input_path, output_path, status, error_type, timestamp`
- **Repo is public (hard assumption)**: Colab `git clone` / `git pull` needs no token / PAT and runs directly.
  **Alternative path (if switched to private)**: clone with a token read from Colab secrets, never hardcoded in the notebook:
  ```python
  from google.colab import userdata
  tok = userdata.get('GH_TOKEN')
  !git clone https://{tok}@github.com/<user>/FinDocStructRAG.git /content/FinDocStructRAG
  ```
  Public for now, but write this fallback path ahead of time so a later switch to private does not block.

### Phase Output Separation

Each phase produces separate artifacts and metrics:

- Phase 1A outputs topology artifacts only:
  - `outputs/tables/gt_filled/`
  - `outputs/tables/tatr_predicted/`
  - topology metrics and screenshots

- Phase 1B outputs OCR-filled extraction artifacts:
  - `outputs/tables/ocr_filled/`
  - content metrics
  - end-to-end extraction QA

- Phase 1C outputs table-only RAG artifacts:
  - `outputs/rag_index/`
  - `outputs/evaluation/rag/`
  - retrieved evidence and grounded QA answers

GT-filled tables are used only for QA pipeline validation and must not be reported as extraction outputs.

### LLM usage guideline (single provider, swappable)

The project is about table extraction + OCR + layout + structure-aware RAG, **not an LLM
comparison**. Therefore:

- **Retrieval has no LLM at any point**: BM25 + dense BGE cosine + RRF + type-aware reranking. HyDE / LLM rewriting / cross-encoder are future work.
- **LLM is used only in answer generation**: after retrieving evidence, generate an answer from that evidence.
- **The MVP wires up only one API LLM** (decided: OpenRouter, an OpenAI-compatible gateway; default model `openai/gpt-4o-mini`), not several providers at once, to avoid an explosion in key management / prompt differences / response format / cost tracking / eval / debug complexity. OpenRouter also lets the model be swapped without changing the SDK.
- **Swappability comes from the wrapper (strict boundary)**: all provider differences (prompt templates, response parsing, retry, cost logging) are contained in `src/llm_client.py`, exposing only `generate_answer()` to the RAG pipeline; switching provider changes only `llm_client.py` + config `LLM_PROVIDER`, not the RAG pipeline.
- **Eval must not touch the SDK**: RAG evaluation and prompt format **may only consume the provider-neutral `LLMAnswer`**, never the SDK's raw response object directly. This way switching provider requires no eval pipeline changes.

```python
# src/llm_client.py
class LLMAnswer(TypedDict):
    answer: str
    cited_evidence_ids: list[str]   # ids of retrieved chunks (for grounding evaluation)
    abstained: bool                 # maps to the V9 unanswerable category

class LLMClient:
    def generate_answer(self, question: str, evidence: list[dict]) -> LLMAnswer: ...

# config.py
LLM_PROVIDER = "openrouter"   # single source of truth switch (OpenAI-compatible gateway)
LLM_MODEL = "openai/gpt-4o-mini"   # swap the model without touching the SDK
```

Implementation details:
1. Return `LLMAnswer` (structured, with citations and an abstain flag), not just a `str` — so grounding/faithfulness evaluation lines up.
2. The prompt must allow abstaining: if evidence is insufficient, answer "cannot answer" and set `abstained=True` (maps to the 5-10 V9 unanswerable questions).
3. Use `temperature=0` at eval time so QA exact match / numeric relaxed match are reproducible.
4. LLM-as-Judge (V9 lists it as optional), if done, goes through the same `LLMClient`, not a separate provider path.
5. API keys use env vars (Colab uses userdata/secrets), never go into git; LLM calls need no GPU and run on either local or Colab.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Colab kernel cannot see local `src/` | notebook starts with `git pull` onto the VM + add the repo path to `sys.path` |
| Local<->Colab code divergence | Git as single source of truth; Colab only `git pull` + runs, never edits `src/` |
| Colab T4 memory / session timeout | outputs land on Drive; process in batches; resumable notebooks |
| FinTabNet.c annotation format mismatch | the first Phase 1A step `can_convert_to_canonical` gate blocks it up front |
| Hard to install torch/paddleocr locally on Windows | requirements split into three; locally install only `requirements-core` + `requirements-dev` |
| Scope too large (multi-week project) | strict sequence Phase 0->1A->1B->1C, ship a demonstrable loop before expanding |
| The V9 `.md` is mojibake | first action when starting is to re-save a readable version in correct UTF-8 |

---

## 6. Milestone checkpoints

| Phase | First delivery | Demonstrable result |
|-------|:--------------:|---------------------|
| 0 | Yes | repo skeleton + 4 smoke tests green |
| 1A | Yes | table topology reconstruction + metrics + 9 screenshots |
| 1B |  | OCR content extraction + content metrics |
| 1C |  | **table-only RAG end-to-end demo loop** |
| 2 |  | DocLayNet layout + table crop integration |
| 3 |  | FUNSD relation P/R/F1 |
| 4 |  | Gradio demo + full eval + report |

---

## 7. Next steps

**Phases 0 through 1C are complete and merged** (v1 = table-only RAG). The next track is
**Phase 2 (DocLayNet layout integration)**; do not open a second track in parallel. The
detector is decided (HF DETR-on-DocLayNet, see §3 Phase 2) but its model id is not yet pinned.
Order:

1. **Detector smoke (Colab T4)**: confirm the chosen HF DETR-on-DocLayNet model loads, runs on a
   couple of DocLayNet pages, and exposes `config.id2label`; only then pin `config.LAYOUT_MODEL`.
2. **Phase 2-local first** (CPU, synthetic unit tests, P3): `bbox_utils.py` (coordinate
   conversions, IoU, crop-with-padding, dedup), `layout_parsing.py` (`LAYOUT_LABEL_MAP` +
   `normalize_label` + the sequential/fallback selection over an injected detector),
   `table_detection.py` (the `table-transformer-detection` fallback adapter).
3. **Then Phase 2-colab**: run detection on a fixed DocLayNet subset, layout AP/IoU via
   `pycocotools`, and confirm a detected table crop feeds the existing Phase 1A topology pipeline.

---

### Sources

- [Google Colab is Coming to VS Code — Google Developers Blog](https://developers.googleblog.com/google-colab-is-coming-to-vs-code/)
- [How do I access local files from colab kernel? — googlecolab/colab-vscode#210](https://github.com/googlecolab/colab-vscode/issues/210)
- [Google Colab VS Code Extension — Marketplace](https://marketplace.visualstudio.com/items?itemName=Google.colab)
