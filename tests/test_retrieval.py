"""Retrieval tests (CPU, synthetic) - Phase 1C. BM25 ranking + RRF fusion."""

from src.retrieval import BM25Index, rrf_fuse, tokenize


def test_tokenize_lowercases_alphanumeric():
    assert tokenize("Revenue $13,223 (2018)!") == ["revenue", "13", "223", "2018"]


def _chunks():
    return [
        {"chunk_id": "table:a", "text": "Revenue 2018 = 13,223; 2017 = 10,376"},
        {"chunk_id": "table:b", "text": "Cash and equivalents 2018 = 500"},
        {"chunk_id": "table:c", "text": "Pension obligations 2019 = 99"},
    ]


def test_bm25_ranks_matching_chunk_first():
    idx = BM25Index(_chunks())
    ranked = idx.search("What was Revenue in 2018?", top_k=3)
    assert ranked[0] == "table:a"
    assert set(ranked) == {"table:a", "table:b", "table:c"}


def test_bm25_top_k_limits_results():
    idx = BM25Index(_chunks())
    assert len(idx.search("2018", top_k=1)) == 1


def test_bm25_empty_query_is_deterministic():
    idx = BM25Index(_chunks())
    # No matching terms -> all zero scores -> tie-broken by chunk_id.
    assert idx.search("zzz", top_k=3) == ["table:a", "table:b", "table:c"]


def test_rrf_ranks_item_high_in_all_lists_first():
    fused = rrf_fuse([["a", "b", "c"], ["a", "c", "d"]], top_k=4)
    assert fused[0] == "a"  # rank 0 in both rankings


def test_rrf_dedupes_and_sums_scores():
    fused = rrf_fuse([["x", "y"], ["y", "x"]], top_k=10)
    assert sorted(fused) == ["x", "y"]  # union, no duplicates
