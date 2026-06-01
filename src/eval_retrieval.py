"""Retrieval metrics for table-only RAG (Phase 1C).

Per-question metrics over a ranked chunk_id list vs the question's relevant_chunk_ids:
hit@k, recall@k, MRR@k. Pure CPU, no model. Unanswerable questions (no relevant chunk)
carry no retrieval judgment and are excluded from the retrieval metrics (they are scored
on the answer side - the abstain path - in the QA eval, not here).

With one relevant chunk per question (the source table), recall@k equals hit@k; both are
reported so the metric still reads correctly if multi-relevant questions are added later.
"""

from __future__ import annotations

from statistics import mean


def hit_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    return len(set(ranked[:k]) & rel) / len(rel)


def mrr_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    for i, cid in enumerate(ranked[:k]):
        if cid in rel:
            return 1.0 / (i + 1)
    return 0.0


def aggregate_retrieval(results: list[dict], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Mean hit@k / recall@k / mrr@k over answerable questions.

    results: dicts with "ranked" (chunk_ids) and "relevant" (relevant chunk_ids). Questions
    with no relevant chunk are skipped (unanswerable: judged on the answer side, not here).
    """
    scored = [r for r in results if r.get("relevant")]
    out: dict = {"num_questions": len(results), "num_answerable": len(scored)}
    if not scored:
        return out
    for k in ks:
        out[f"hit@{k}"] = mean(hit_at_k(r["ranked"], r["relevant"], k) for r in scored)
        out[f"recall@{k}"] = mean(recall_at_k(r["ranked"], r["relevant"], k) for r in scored)
        out[f"mrr@{k}"] = mean(mrr_at_k(r["ranked"], r["relevant"], k) for r in scored)
    return out
