# Layout-Aware Financial Document Intelligence Pipeline — Design Spec (V9)

**PDF layout parsing, financial table extraction, OCR/form relation linking, and retrieval-grounded QA on Colab T4.**

> This is the **design spec**: it defines *what* to build (schemas, function signatures,
> metrics, tests). For the *order, environment workflow, and acceptance criteria*, see the
> execution roadmap in [PLAN.md](PLAN.md). Section numbers here are referenced from PLAN.md
> as `§4.2`, `§5.13`, etc., so they are kept stable.

---

## 1. Project positioning

**Production-oriented Document AI prototype.** Under Colab T4 constraints, demonstrate a
complete, evaluable, reproducible document intelligence pipeline.

> This project is a production-oriented Document AI prototype designed to demonstrate layout-aware parsing, financial table extraction, OCR/form relation linking, structured outputs, hybrid retrieval, source-grounded QA, and evaluation under Colab T4 constraints.

---

## 2. Dataset selection

| Module | Dataset | Debug | MVP | Final |
|--------|---------|-------|-----|-------|
| V1 Table | FinTabNet.c | 30-50 | 100-300 | 300-500 |
| V2 Layout | DocLayNet | 30-50 | 100-300 | 300-500 |
| V3 OCR/Form | FUNSD | 20 | 199 | 199 |
| QA | self-built | 10 | 30-50 | 50-100 |

FinTabNet.c: S&P 500 annual reports; Phase 1 uses cropped table images.
DocLayNet: `unique("doc_category")` first, then filter.
FUNSD: V1 = relation-linking baseline over GT tokens/entities.
QA: table lookup 15-20 / cross-table 15-20 / form field 10-15 / unanswerable 5-10.

---

## 3. Pipeline architecture

```
PDF / image / scan -> Layout parsing -> Table detection + structure recognition
-> OCR / form relation linking -> Structured output (JSON/CSV/HTML/MD)
-> Metadata-preserving chunking -> Hybrid retrieval + source-grounded QA
-> Evaluation + error analysis -> Demo + final report
```

---

## 4. Key architecture decisions

### 4.1 Sequential-first with fallback
Layout detection -> table region -> bbox_utils crop -> TATR -> postprocess -> fallback if low confidence -> IoU dedup.

### 4.2 Table detection vs structure recognition

| Task | Model |
|------|-------|
| Detection (Phase 2 fallback) | `microsoft/table-transformer-detection` |
| Structure recognition | `microsoft/table-transformer-structure-recognition-v1.1-fin` |

### 4.3 Bbox coordinate mapping
`src/bbox_utils.py`, the first step of Phase 2.

---

## 5. TATR post-processing

### 5.1 src/tatr_postprocess.py

```python
boxes_to_grid()                  # row x col -> cell bbox derivation
validate_grid_geometry()         # grid sanity checks
map_spanning_bbox_to_grid()      # spanning bbox -> grid coordinates
apply_spanning_cells()           # merge spanning into grid
assign_cells_to_rows_columns()
detect_headers()
assign_words_to_cells()          # OCR words -> cells
export_html()
export_csv()
validate_dataframe()
normalize_table_annotation()     # GT -> canonical
normalize_tatr_prediction()      # pred -> canonical
html_to_canonical()              # occupancy-aware HTML parser
```

### 5.2 Cell bbox derivation

```python
def boxes_to_grid(row_boxes, col_boxes, spanning_cells=None):
    """
    cell_bbox = (col.x1, row.y1, col.x2, row.y2)
    Spanning cells override via map_spanning_bbox_to_grid().
    """
    rows = sorted(row_boxes, key=lambda r: r["bbox"][1])
    cols = sorted(col_boxes, key=lambda c: c["bbox"][0])

    cells = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            cells.append({
                "row_start": i, "row_end": i + 1,
                "col_start": j, "col_end": j + 1,
                "bbox": [col["bbox"][0], row["bbox"][1],
                         col["bbox"][2], row["bbox"][3]],
                "text": "", "is_header": False, "words": []
            })

    if spanning_cells:
        cells = apply_spanning_cells(cells, spanning_cells, rows, cols)

    return cells
```

### 5.3 Grid geometry validation

```python
def validate_grid_geometry(row_boxes, col_boxes, cells, logger=None):
    """
    Checks: negative dims, sort order, adjacent overlap > 0.3,
    tiny cells (area < 100).
    """
```

### 5.4 Spanning cell grid mapping (added in V9)

A TATR-predicted spanning cell is a single bbox; it must be mapped back to grid coordinates
to be evaluated and merged.

```python
def map_spanning_bbox_to_grid(spanning_bbox, rows, cols, overlap_threshold=0.5):
    """
    Convert predicted spanning cell bbox into grid coordinates
    (row_start, row_end, col_start, col_end) by computing overlap
    with each row/column box.

    A row/col is considered covered if:
        overlap_length / row_or_col_length >= overlap_threshold

    Args:
        spanning_bbox: [x1, y1, x2, y2] of predicted spanning cell
        rows: sorted list of row box dicts with "bbox" key
        cols: sorted list of col box dicts with "bbox" key
        overlap_threshold: minimum overlap ratio to consider covered

    Returns:
        {
            "row_start": int, "row_end": int,  # exclusive
            "col_start": int, "col_end": int,  # exclusive
            "bbox": [x1, y1, x2, y2]           # original spanning bbox
        }
        or None if no rows/cols meet threshold
    """
    sx1, sy1, sx2, sy2 = spanning_bbox

    # Find covered rows
    covered_rows = []
    for i, row in enumerate(rows):
        ry1, ry2 = row["bbox"][1], row["bbox"][3]
        overlap = max(0, min(sy2, ry2) - max(sy1, ry1))
        row_height = ry2 - ry1
        if row_height > 0 and overlap / row_height >= overlap_threshold:
            covered_rows.append(i)

    # Find covered columns
    covered_cols = []
    for j, col in enumerate(cols):
        cx1, cx2 = col["bbox"][0], col["bbox"][2]
        overlap = max(0, min(sx2, cx2) - max(sx1, cx1))
        col_width = cx2 - cx1
        if col_width > 0 and overlap / col_width >= overlap_threshold:
            covered_cols.append(j)

    if not covered_rows or not covered_cols:
        return None

    return {
        "row_start": min(covered_rows),
        "row_end": max(covered_rows) + 1,
        "col_start": min(covered_cols),
        "col_end": max(covered_cols) + 1,
        "bbox": spanning_bbox
    }


def apply_spanning_cells(cells, spanning_cells, rows, cols):
    """
    For each predicted spanning cell bbox:
    1. Map to grid coordinates via map_spanning_bbox_to_grid()
    2. Remove individual cells covered by the span
    3. Insert merged spanning cell
    """
    for span_box in spanning_cells:
        mapped = map_spanning_bbox_to_grid(span_box["bbox"], rows, cols)
        if mapped is None:
            continue

        # Remove covered individual cells
        cells = [c for c in cells if not (
            c["row_start"] >= mapped["row_start"] and
            c["row_end"] <= mapped["row_end"] and
            c["col_start"] >= mapped["col_start"] and
            c["col_end"] <= mapped["col_end"]
        )]

        # Insert spanning cell
        cells.append({
            "row_start": mapped["row_start"],
            "row_end": mapped["row_end"],
            "col_start": mapped["col_start"],
            "col_end": mapped["col_end"],
            "bbox": mapped["bbox"],
            "text": "", "is_header": False, "words": []
        })

    return cells
```

`spanning_cell_detection_rate` depends on this function: a predicted spanning bbox is first
mapped to grid coordinates so it can be compared position-by-position against GT spanning
cells.

### 5.5 Word source strategy

Phase 1A: GT cell text (topology only). Phase 1B: OCR (PaddleOCR > Tesseract).

### 5.6 Phase 1A QA disclaimer

GT-filled QA = pipeline validation. End-to-end QA = Phase 1B only.

### 5.7 Phase 1A: GT-filled vs TATR-predicted, separate outputs

```
outputs/tables/
├── gt_filled/          <- QA validation
├── tatr_predicted/     <- topology evaluation
└── failures/
```

Metadata records `text_source` + `evaluation_type`. GT text is never used as an extraction
output.

### 5.8 Canonical table schema

`normalize_table_annotation()` + `normalize_tatr_prediction()` -> the same canonical schema.

### 5.9 FinTabNet.c annotation format gate

Notebook 01, top priority: the `can_convert_to_canonical` gate.

### 5.10 Occupancy-aware HTML parser

```python
def html_to_canonical(html_str: str) -> dict:
    """Uses occupancy grid (occupied set) for rowspan/colspan."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_str, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return {"num_rows": 0, "num_cols": 0, "cells": []}

    occupied = set()
    cells = []
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for td in row.find_all(["td", "th"]):
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            rowspan = int(td.get("rowspan", 1))
            colspan = int(td.get("colspan", 1))
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied.add((r, c))
            cells.append({
                "row_start": row_idx, "row_end": row_idx + rowspan,
                "col_start": col_idx, "col_end": col_idx + colspan,
                "text": td.get_text(strip=True),
                "is_header": td.name == "th"
            })
            col_idx += colspan

    num_rows = len(rows)
    num_cols = max((c["col_end"] for c in cells), default=0)
    return {"num_rows": num_rows, "num_cols": num_cols, "cells": cells}
```

### 5.11 OCR output schema

`OCRWord` dataclass + `paddleocr_to_words()` / `tesseract_to_words()`. PaddleOCR priority.

### 5.12 `assign_words_to_cells()` matching rule

Center-in-cell -> max IoU fallback -> unassigned logging -> sort by y,x -> join.

### 5.13 Numeric normalization (V9 fixes `looks_numeric()`)

V8's `looks_numeric()` used `[\d$£€¥%(),.\-–—]`, which made `"Operating Income (Loss)"` count
as numeric because of the parentheses, triggering O->0 / I->1 substitutions.

**V9 fix: require at least one digit, or a pure dash:**

```python
# src/numeric_utils.py

import re
from typing import Optional


def looks_numeric(raw: str) -> bool:
    """
    Conservative: must contain at least one digit, or be a pure dash.
    Parentheses alone do NOT make a string numeric.
    """
    s = raw.strip()
    if s in ("-", "–", "—"):
        return True
    return bool(re.search(r'\d', s))


def normalize_financial_number(
    raw: str,
    dash_as_zero: bool = True,
    percent_as_ratio: bool = True
) -> Optional[float]:
    """
    Normalize financial number string to float.
    Returns None if not numeric.

    dash_as_zero: dash -> 0.0 (True) or None (False)
    percent_as_ratio: 12.5% -> 0.125 (True) or 12.5 (False)
    """
    if not raw or not raw.strip():
        return None

    s = raw.strip()
    s = re.sub(r'[$£€¥]', '', s).strip()

    if s in ('-', '–', '—', '- ', ' -'):
        return 0.0 if dash_as_zero else None

    is_negative = False
    if s.startswith('(') and s.endswith(')'):
        is_negative = True
        s = s[1:-1].strip()

    is_percent = False
    if s.endswith('%'):
        is_percent = True
        s = s[:-1].strip()

    # OCR substitutions — ONLY if string contains a digit
    if looks_numeric(s):
        s = s.replace('O', '0').replace('o', '0')
        s = s.replace('l', '1').replace('I', '1')

    s = s.replace(' ', '').replace(',', '')

    try:
        value = float(s)
    except ValueError:
        return None

    if is_negative:
        value = -value
    if is_percent and percent_as_ratio:
        value = value / 100.0

    return value


def relaxed_numeric_match(
    pred: str, gt: str,
    tolerance: float = 0.01,
    dash_as_zero: bool = True,
    percent_as_ratio: bool = True
) -> bool:
    pred_val = normalize_financial_number(pred, dash_as_zero, percent_as_ratio)
    gt_val = normalize_financial_number(gt, dash_as_zero, percent_as_ratio)
    if pred_val is None or gt_val is None:
        return False
    if gt_val == 0:
        return abs(pred_val) < 1e-6
    return abs(pred_val - gt_val) / abs(gt_val) < tolerance
```

**`looks_numeric()` behavior comparison:**

| Input | V8 | V9 | Correct? |
|-------|----|----|----------|
| `"1,234"` | True | True | Yes |
| `"(1,234)"` | True | True | Yes (has digit) |
| `"$1,234"` | True | True | Yes |
| `"—"` | True | True | Yes |
| `"12.5%"` | True | True | Yes |
| `"Operating Income (Loss)"` | True (wrong) | **False** (correct) | Fixed in V9 |
| `"Total Assets"` | False | False | Yes |
| `"1O,234"` | True | True | Yes (has digit) |

### 5.14 Phase 1A deliverable screenshots

```
1. Original table crop image
2. TATR row/column/header boxes overlay
3. Derived cell grid overlay (row x col intersection)
4. Spanning cell mapping visualisation (if detected)
5. Grid geometry validation report
6. Reconstructed HTML table (GT-filled, from gt_filled/)
7. TATR predicted topology table (from tatr_predicted/)
8. Failure case visualisation
9. Topology metrics summary
```

---

## 6. Table evaluation

### 6.1 Three tiers + GriTS

Topology (Phase 1A) -> Content (Phase 1B) -> End-to-end QA -> GriTS (Final/stretch).

### 6.2 Proxy metrics

**Topology:** row/col count accuracy, cell_occupancy_f1, spanning_cell_detection_rate (via `map_spanning_bbox_to_grid`), header_detection_accuracy, parse/html_success_rate, html_structure_match.

**Content:** cell_text_exact_match, numeric_cell_relaxed_match (via `numeric_utils`), non_empty_cell_content_f1.

**QA:** qa_exact_match, qa_numeric_relaxed_match (1%).

### 6.3 Phase 1A vs 1B

| Metric | 1A | 1B |
|--------|----|----|
| Topology | Yes | Yes |
| Content | No | Yes |
| QA pipeline validation (disclaimer) | Yes | — |
| QA end-to-end | — | Yes |

---

## 7. RAG

BM25 + FAISS + RRF + type-aware reranking + query routing (rule-based) + source grounding +
`normalize_financial_number()` cross-check. HyDE / LLM rewriting / cross-encoder = future
work. LLM-as-Judge = optional.

Dense embedding for the FAISS index: **`BAAI/bge-small-en-v1.5`** (locked in
[PLAN.md](PLAN.md) §0; V9 left it unspecified).

Structure-aware chunking (text/table/KV/header routing). Table serialization experiment
(Markdown vs linearized). 5 error categories for RAG evaluation.

### 7.1 Answer generation (single provider, swappable)

Retrieval has no LLM at any point. The LLM is used **only** to generate an answer from
retrieved evidence, behind the `src/llm_client.py` abstraction. The MVP wires up a single
provider (**Gemini**); switching provider changes only `llm_client.py` + config
`LLM_PROVIDER`, never the RAG pipeline. (Locked in [PLAN.md](PLAN.md) §4; V9 did not specify
how the grounded answer is produced.)

```python
# src/llm_client.py
from typing import TypedDict


class LLMAnswer(TypedDict):
    answer: str
    cited_evidence_ids: list[str]   # ids of retrieved chunks (for grounding evaluation)
    abstained: bool                 # maps to the unanswerable QA category (§2)


class LLMClient:
    def generate_answer(self, question: str, evidence: list[dict]) -> LLMAnswer: ...

# config.py
LLM_PROVIDER = "gemini"   # single source of truth switch
```

Rules:
- The prompt must allow abstaining: if evidence is insufficient, set `abstained=True` (maps to the 5-10 unanswerable QA items).
- Eval uses `temperature=0` for reproducible exact match / numeric relaxed match.
- Evaluation consumes only the provider-neutral `LLMAnswer`, never the SDK raw response.
- API keys come from env vars (Colab uses userdata/secrets), never committed to git.

---

## 8. Evaluation library vs custom

| Module | Approach |
|--------|----------|
| Layout AP/IoU | pycocotools |
| Table MVP | custom proxy (spanning via `map_spanning_bbox_to_grid`) |
| Table Final | GriTS |
| FUNSD relation V1 | custom + sklearn |
| FUNSD token V2 | seqeval |
| Retrieval | ranx |
| Numeric | src/numeric_utils.py |
| RAG faithfulness | Ragas/DeepEval (optional) |
| Tests | pytest |

---

## 9. DocLayNet label mapping

`model.config.id2label` -> `LAYOUT_LABEL_MAP` -> `normalize_label()`.

---

## 10. Failure logging

Schema + error type taxonomy + `src/failure_logger.py`.

---

## 11. Incremental milestones

| Phase | Content | Redo? |
|-------|---------|-------|
| 0 | Repo skeleton | first |
| 1A | Table topology (GT text, gt_filled/ + tatr_predicted/) | first |
| 1B | OCR word assignment + content metrics + end-to-end QA | no |
| 2 | DocLayNet layout + table crop + sequential pipeline | no |
| 3 | FUNSD relation branch | no |
| 4 | Eval + demo + report | no |

### Phase 1A (~3-4 days)
gt_format_report gate -> `html_to_canonical()` (occupancy-aware) -> `boxes_to_grid()` + `validate_grid_geometry()` + `map_spanning_bbox_to_grid()` -> gt_filled/ + tatr_predicted/ -> topology metrics -> QA pipeline validation (disclaimer) -> 9 screenshots -> failure logging.

### Phase 1B (~2-3 days)
PaddleOCR -> OCRWord -> `assign_words_to_cells()` -> `numeric_utils` (V9 `looks_numeric`) -> content metrics -> end-to-end QA. Push GitHub V1.

### Phase 1C (~2-3 days, added in PLAN.md)
Table-only RAG QA: build table chunks -> BM25 + FAISS (`BAAI/bge-small-en-v1.5`) + RRF -> Markdown vs linearized serialization experiment -> source-grounded QA via `src/llm_client.py` -> report GT-filled QA vs OCR-filled QA separately. This is the v1 release (see PLAN.md §2).

### Phase 2 (~1-1.5 weeks)
bbox_utils + label mapping -> DocLayNet -> sequential + fallback.

### Phase 3 (~1 week)
FUNSD spatial heuristic + boosts -> relation P/R/F1.

### Phase 4 (~2-3 days)
Gradio + full eval + report.

---

## 12. Folder structure

```
FinDocStructRAG/
├── README.md
├── requirements.txt              # local entry point: -r core + -r dev
├── requirements-core.txt         # local dev (pandas, bs4, numpy, lxml, LLM SDK)
├── requirements-colab.txt        # GPU (torch, transformers, paddleocr, faiss, datasets)
├── requirements-dev.txt          # tooling (pytest, ruff, black, mypy)
├── .gitignore
├── notebooks/ (00–09)            # Colab runners + demo/report only (no logic)
│
├── scripts/                      # repeatable pipeline runners (CLI)
│   ├── run_phase1a_local.py
│   ├── run_phase1a_colab.py
│   ├── build_table_chunks.py
│   ├── evaluate_tables.py
│   └── evaluate_rag.py
│
├── src/
│   ├── __init__.py, config.py, data_utils.py
│   ├── bbox_utils.py
│   ├── table_detection.py
│   ├── tatr_postprocess.py      <- boxes_to_grid, validate_grid_geometry,
│   │                               map_spanning_bbox_to_grid, apply_spanning_cells,
│   │                               html_to_canonical (occupancy), canonical schema
│   ├── table_extraction.py
│   ├── layout_parsing.py        <- normalize_label
│   ├── ocr_adapter.py
│   ├── numeric_utils.py         <- V9: fixed looks_numeric, configurable
│   ├── funsd_extraction.py
│   ├── chunking.py, retrieval.py, query_router.py, qa.py
│   ├── llm_client.py            <- single-provider answer generation, LLMAnswer
│   ├── failure_logger.py
│   ├── eval_layout.py, eval_table.py, eval_funsd.py
│   ├── eval_retrieval.py, eval_rag.py, eval_runtime.py
│   └── visualisation.py
│
├── tests/
│   ├── test_bbox_utils.py
│   ├── test_tatr_postprocess.py     <- +spanning mapping/apply tests
│   ├── test_funsd_relations.py
│   ├── test_chunk_schema.py
│   ├── test_label_mapping.py
│   ├── test_ocr_adapter.py
│   └── test_numeric_utils.py        <- +looks_numeric_parentheses_no_digits
│
├── data/ (raw/ processed/ samples/)
├── outputs/ (layout/ tables/{gt_filled,tatr_predicted,ocr_filled,failures}/ funsd/
│             integrated/ rag_index/ evaluation/ failure_logs/ manifests/)
├── assets/
└── reports/
```

---

## 13. Tests

```
test_tatr_postprocess.py adds:
  test_map_spanning_bbox_covers_two_rows()
  test_map_spanning_bbox_covers_three_cols()
  test_map_spanning_bbox_no_overlap_returns_none()
  test_apply_spanning_cells_merges_correctly()
  test_apply_spanning_cells_removes_covered()

test_numeric_utils.py adds/fixes:
  test_looks_numeric_with_digits()             # "1,234" -> True
  test_looks_numeric_pure_dash()               # "—" -> True
  test_looks_numeric_parentheses_no_digits()   # "Operating Income (Loss)" -> False
  test_looks_numeric_parentheses_with_digits() # "(1,234)" -> True
  test_looks_numeric_plain_text()              # "Total Assets" -> False
  test_ocr_sub_not_applied_to_text()           # "Operating" stays "Operating"
```

---

## 14. README limitations

> This project is a production-oriented prototype using subset evaluation.
>
> Table evaluation separates topology from content metrics. Cell bboxes are derived from row/column intersections; spanning cells are mapped back to grid coordinates via overlap-ratio thresholding. Grid geometry is validated for overlaps and degenerate cells. Phase 1A produces GT-filled and TATR-predicted tables as separate outputs.
>
> Financial number normalization requires at least one digit to trigger OCR character substitutions, preventing false positives on text like "Operating Income (Loss)". Dash-as-zero and percent-as-ratio conventions are configurable.
>
> FUNSD V1 uses GT tokens/entities for relation-linking. RAG uses BM25+FAISS, RRF, rule-based routing, type-aware reranking, source grounding, numeric validation. HyDE, cross-encoder, LLM rewriting are future work.

---

## 15. Resume bullets

**Short version:**

Built a production-oriented Document AI prototype for PDFs, financial tables, and OCR/form outputs using FinTabNet.c, DocLayNet, and FUNSD. The system reconstructs table structures from row/column detection with grid validation and spanning cell mapping, assigns OCR words to derived cells, normalizes financial numbers, and supports retrieval-grounded QA with source citations.

**Technical version:**

Developed a Colab-compatible Document AI pipeline combining Table Transformer structure recognition with cell bbox derivation from row/column intersections, spanning cell bbox-to-grid mapping, grid geometry validation, occupancy-aware HTML parsing, OCR word-to-cell assignment with conservative financial number normalization, DocLayNet layout parsing with label normalization, FUNSD relation-linking baseline, structure-aware chunking, BM25/FAISS hybrid retrieval with RRF and type-aware reranking, and source-grounded QA. Topology and content evaluation are reported separately.

---

## 16. V8 -> V9 diff summary

| Item | V8 | V9 |
|------|----|----|
| `looks_numeric()` | `[\d$£€¥%(),.\-–—]` — parentheses trigger it | **must contain a digit, or be a pure dash** |
| `"Operating Income (Loss)"` | True -> OCR sub corrupts it | **False -> not triggered** |
| Spanning cell mapping | `apply_spanning_cells()` mapping undefined | **added `map_spanning_bbox_to_grid()` overlap-ratio mapping** |
| `apply_spanning_cells()` | not implemented | **complete: map -> remove covered -> insert merged** |
| `spanning_cell_detection_rate` | hard to compute reliably | **relies on `map_spanning_bbox_to_grid()` position-level comparison** |
| Deliverable screenshots | 8 | **9 (+spanning cell mapping vis)** |
| tests/ | — | **+spanning mapping tests, +looks_numeric parentheses** |

---

## 17. Supplements beyond V9 (alignment with PLAN.md)

These items were not in the original V9 spec and were added during planning. They do not
change V9's table/OCR/eval design; they fill gaps V9 left open:

- **Embedding model** for the FAISS dense index: `BAAI/bge-small-en-v1.5` (§7). V9 specified hybrid retrieval but no embedding model.
- **Answer-generation contract** (§7.1): a single-provider, swappable `src/llm_client.py` returning a provider-neutral `LLMAnswer`. V9 mentioned source-grounded QA but not how the answer is generated or which LLM.
- **Phase 1C** (§11): table-only RAG QA inserted between content extraction and the full pipeline, making the first demonstrable RAG loop the v1 release instead of waiting for Phase 4.
- **Repo layout** (§12): `scripts/` for repeatable runners, three-way requirements split, `src/llm_client.py`, and `outputs/manifests/` for resumable batch runs.

For the order, environment workflow (VS Code + Colab extension, git-as-truth, `.py`-core /
`.ipynb`-runner), and acceptance criteria, see [PLAN.md](PLAN.md).
