# QA evaluation set (Phase 1C)

The table-only RAG eval set is **hybrid**:

- **templated** (auto) - lookup / numeric questions generated from GT cells by
  `scripts/build_qa_dataset.py`; answer = the GT cell value, relevance = the source table.
  You do not author these.
- **manual + unanswerable** (this folder) - the harder questions templating cannot produce:
  cross-cell comparison, reasoning, paraphrase, and plausible-but-unanswerable questions.
  **You author these by hand** in `qa_manual_seed.jsonl`.

The two are merged into `outputs/qa/qa_all.jsonl` at build time. Gold answers always come
from GT; the eval runs over the GT-filled and OCR-filled corpora separately (P4).

## How to author the manual set

1. Browse real tables you can read the true answer off:
   ```
   %run scripts/preview_chunks.py --limit 25 --seed 7 --format image --display
   ```
   It displays each table as a PNG with its `sample_id` and `chunk_id`. If you run the
   script with `!python` instead of `%run`, it still writes the PNG files and `index.html`
   under `outputs/figures/phase1c_preview/`, but inline notebook display is unavailable.
2. For ~10 harder questions and ~5-10 unanswerable ones, add one JSON object per line to
   `qa_manual_seed.jsonl`, using a real `sample_id` you just read.

## Authoring rules

1. **Disambiguate by natural context, never by leaking ids.** Name enough of the table's
   real content (its title and row/column labels) that the question points to exactly one
   table in the corpus - but do not print the `sample_id` / `chunk_id` in the question text.
2. **An unanswerable question must be unanswerable everywhere.** Phrase it as a field
   genuinely absent from the named table's schema, not one another table in the corpus
   happens to contain (e.g. "accrued marketing expense" in an accrued-liabilities table that
   has no such row). A question another table could answer is mislabeled, not unanswerable.
3. **Match the gold answer to how it is scored.** `numeric_relaxed` keys off
   `looks_numeric(gold_answer)`, not `answer_type`: a question you intend to score
   numerically needs a single numeric gold (one number, and no `%` sign when the gold is a
   plain figure). Avoid brittle yes/no golds - they are scored as exact strings and break on
   paraphrase.

## Record schema (one JSON object per line)

| field | meaning |
| --- | --- |
| `question_id` | unique id; use `mq_0001..` (manual), `uq_0001..` (unanswerable) |
| `question` | the question text |
| `gold_answer` | the correct answer read off the GT table; `""` if unanswerable |
| `answer_type` | `numeric`, `text`, or `unanswerable` |
| `sample_id` | the table the answer comes from (the stem, e.g. `IP_2012_page_114_table_2`) |
| `relevant_chunk_ids` | `["table:<sample_id>"]`; `[]` if unanswerable |
| `source` | `manual` or `manual_unanswerable` |
| `is_answerable` | `true` / `false` |

## Examples

Answerable (numeric cross-cell reasoning, pinned by table content):
```json
{"question_id":"mq_0001","question":"In the table headed Years ended December 31 with rows Commercial Airplanes and Boeing Capital Corporation, how much larger was Commercial Airplanes than Boeing Capital Corporation in 2009?","gold_answer":"$285","answer_type":"numeric","sample_id":"BA_2009_page_122_table_0","relevant_chunk_ids":["table:BA_2009_page_122_table_0"],"source":"manual","is_answerable":true}
```

Unanswerable (a field absent from the named table, not answerable by any other table):
```json
{"question_id":"uq_0001","question":"In the accrued liabilities table for Sep 30, 2012 and Oct 2, 2011, what was accrued marketing expense at Sep 30, 2012?","gold_answer":"","answer_type":"unanswerable","sample_id":"SBUX_2012_page_78_table_1","relevant_chunk_ids":[],"source":"manual_unanswerable","is_answerable":false}
```

Verify the answer against the `preview_chunks.py` image/table preview before committing a
row - these are eval ground truth.
