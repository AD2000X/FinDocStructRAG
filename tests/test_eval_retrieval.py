"""Retrieval metrics tests (CPU, synthetic) - Phase 1C."""

from src.eval_retrieval import aggregate_retrieval, hit_at_k, mrr_at_k, recall_at_k


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], ["b"], 1) == 0.0
    assert hit_at_k(["a", "b", "c"], ["b"], 2) == 1.0


def test_recall_at_k_single_relevant():
    assert recall_at_k(["a", "b"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "b"], ["c"], 2) == 0.0


def test_mrr_at_k_uses_first_relevant_rank():
    assert mrr_at_k(["a", "b", "c"], ["c"], 3) == 1.0 / 3
    assert mrr_at_k(["a", "b", "c"], ["c"], 2) == 0.0  # outside k


def test_aggregate_skips_unanswerable():
    results = [
        {"ranked": ["a", "b"], "relevant": ["a"]},   # hit@1 = 1
        {"ranked": ["x", "y"], "relevant": ["y"]},   # hit@1 = 0, mrr = 0.5
        {"ranked": ["p", "q"], "relevant": []},      # unanswerable -> skipped
    ]
    agg = aggregate_retrieval(results, ks=(1, 2))
    assert agg["num_questions"] == 3
    assert agg["num_answerable"] == 2
    assert agg["hit@1"] == 0.5
    assert agg["hit@2"] == 1.0
    assert agg["mrr@2"] == (1.0 + 0.5) / 2


def test_aggregate_no_answerable_returns_counts_only():
    agg = aggregate_retrieval([{"ranked": ["a"], "relevant": []}], ks=(1,))
    assert agg["num_answerable"] == 0
    assert "hit@1" not in agg
