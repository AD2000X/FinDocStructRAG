"""QA answer metrics for table-only RAG (Phase 1C).

Scores an LLMAnswer against the gold QA record. Pure CPU, no model. Reuses the same
text/number normalization as the content eval (numeric_utils) so an answer is judged the
same way a cell is, not by a second comparator.

Metrics over answerable questions:
- answer_exact: normalized exact-string match (whitespace + dot-leaders, casefolded).
- numeric_relaxed: relaxed numeric match (over questions whose gold is numeric); catches
  "$495" == "495".
- citation_hit: the model cited at least one relevant (gold source) chunk.
- abstain_rate: how often the model abstained on an answerable question (a diagnostic;
  lower is better).

For unanswerable questions (the hand-authored set, added later) the only judgment is
abstain_accuracy = how often the model correctly abstained.
"""

from __future__ import annotations

from statistics import mean

from .numeric_utils import looks_numeric, normalize_cell_text, relaxed_numeric_match


def answer_exact_match(pred: str, gold: str) -> float:
    return 1.0 if (normalize_cell_text(pred).casefold()
                   == normalize_cell_text(gold).casefold()) else 0.0


def answer_numeric_relaxed(pred: str, gold: str) -> float:
    return 1.0 if relaxed_numeric_match(pred, gold) else 0.0


def citation_hit(citations: list[str], relevant: list[str]) -> float:
    return 1.0 if set(citations) & set(relevant) else 0.0


def aggregate_qa(results: list[dict]) -> dict:
    """Aggregate per-question QA results.

    results: dicts with pred (str), gold (str), citations (list), relevant (list),
    abstained (bool), is_answerable (bool). Answerable and unanswerable are scored separately.
    """
    answerable = [r for r in results if r.get("is_answerable", True)]
    unanswerable = [r for r in results if not r.get("is_answerable", True)]
    out: dict = {
        "num_questions": len(results),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
    }
    if answerable:
        out["answer_exact"] = mean(
            answer_exact_match(r["pred"], r["gold"]) for r in answerable)
        numeric = [r for r in answerable if looks_numeric(r["gold"])]
        out["numeric_relaxed"] = (
            mean(answer_numeric_relaxed(r["pred"], r["gold"]) for r in numeric)
            if numeric else None)
        out["num_numeric"] = len(numeric)
        out["citation_hit"] = mean(
            citation_hit(r["citations"], r["relevant"]) for r in answerable)
        out["abstain_rate"] = mean(1.0 if r["abstained"] else 0.0 for r in answerable)
    if unanswerable:
        out["abstain_accuracy"] = mean(
            1.0 if r["abstained"] else 0.0 for r in unanswerable)
    return out
