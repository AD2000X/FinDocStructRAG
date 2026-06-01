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
   !python scripts/preview_chunks.py --limit 15 --seed 7
   ```
   It prints each table's `sample_id`, `chunk_id`, and a markdown rendering.
2. For ~10 harder questions and ~5-10 unanswerable ones, add one JSON object per line to
   `qa_manual_seed.jsonl`, using a real `sample_id` you just read.

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

Answerable (cross-cell comparison):
```json
{"question_id":"mq_0001","question":"Did Net Sales rise from 2011 to 2012?","gold_answer":"Yes","answer_type":"text","sample_id":"IP_2012_page_114_table_2","relevant_chunk_ids":["table:IP_2012_page_114_table_2"],"source":"manual","is_answerable":true}
```

Unanswerable (not present in the table):
```json
{"question_id":"uq_0001","question":"What was the operating margin?","gold_answer":"","answer_type":"unanswerable","sample_id":"IP_2012_page_114_table_2","relevant_chunk_ids":[],"source":"manual_unanswerable","is_answerable":false}
```

Verify the answer against the markdown `preview_chunks.py` prints before committing a row -
these are eval ground truth.
