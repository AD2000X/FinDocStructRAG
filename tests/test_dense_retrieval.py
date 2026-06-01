"""Dense retrieval tests (CPU, synthetic) - Phase 1C.

A toy bag-of-words embedder over a fixed vocab exercises the same ranking path as BGE
without loading any model, so the index + tie-break logic are verified locally (P3).
"""

import numpy as np

from src.dense_retrieval import DenseIndex, rank_chunk_ids

_VOCAB = ["revenue", "cash", "pension", "2018", "2019"]


def _toy_embedder(texts):
    """One-hot-ish bag of words over _VOCAB; shared query/passage embedder (no instruction)."""
    rows = []
    for t in texts:
        toks = set(t.lower().split())
        rows.append([1.0 if w in toks else 0.0 for w in _VOCAB])
    return np.array(rows, dtype="float32")


def _chunks():
    return [
        {"chunk_id": "table:a", "text": "revenue 2018"},
        {"chunk_id": "table:b", "text": "cash 2018"},
        {"chunk_id": "table:c", "text": "pension 2019"},
    ]


def test_rank_chunk_ids_tie_break_by_id():
    ranked = rank_chunk_ids([("b", 1.0), ("a", 1.0), ("c", 0.5)], top_k=3)
    assert ranked == ["a", "b", "c"]  # a and b tie on score -> id order; c last


def test_dense_ranks_matching_chunk_first():
    idx = DenseIndex(_chunks(), _toy_embedder, query_instruction="")
    ranked = idx.search("revenue 2018", top_k=3)
    assert ranked[0] == "table:a"
    assert set(ranked) == {"table:a", "table:b", "table:c"}


def test_dense_top_k_limits_results():
    idx = DenseIndex(_chunks(), _toy_embedder, query_instruction="")
    assert len(idx.search("2018", top_k=1)) == 1


def test_dense_empty_corpus_returns_empty():
    idx = DenseIndex([], _toy_embedder, query_instruction="")
    assert idx.search("revenue", top_k=3) == []
