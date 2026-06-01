"""QA metric tests (CPU, synthetic) - Phase 1C."""

from src.eval_qa import (
    aggregate_qa,
    answer_exact_match,
    answer_numeric_relaxed,
    citation_hit,
)


def test_answer_exact_match_normalizes():
    assert answer_exact_match("13,223", "13,223") == 1.0
    assert answer_exact_match("  Revenue ", "revenue") == 1.0  # whitespace + case
    assert answer_exact_match("13,223", "13,224") == 0.0


def test_answer_numeric_relaxed_currency():
    assert answer_numeric_relaxed("$495", "495") == 1.0   # currency symbol tolerated
    assert answer_numeric_relaxed("495", "600") == 0.0    # outside the 1% relaxed tolerance


def test_citation_hit():
    assert citation_hit(["table:a"], ["table:a"]) == 1.0
    assert citation_hit(["table:b"], ["table:a"]) == 0.0
    assert citation_hit([], ["table:a"]) == 0.0


def test_aggregate_qa_answerable():
    results = [
        {"pred": "13,223", "gold": "13,223", "citations": ["table:a"],
         "relevant": ["table:a"], "abstained": False, "is_answerable": True},
        {"pred": "$495", "gold": "495", "citations": ["table:b"],
         "relevant": ["table:a"], "abstained": False, "is_answerable": True},
    ]
    agg = aggregate_qa(results)
    assert agg["num_answerable"] == 2
    assert agg["answer_exact"] == 0.5            # only the first is an exact string match
    assert agg["numeric_relaxed"] == 1.0         # both numeric values match
    assert agg["citation_hit"] == 0.5            # second cites the wrong table
    assert agg["abstain_rate"] == 0.0


def test_aggregate_qa_unanswerable_scores_abstention():
    results = [
        {"pred": "", "gold": "", "citations": [], "relevant": [],
         "abstained": True, "is_answerable": False},
        {"pred": "made up", "gold": "", "citations": [], "relevant": [],
         "abstained": False, "is_answerable": False},
    ]
    agg = aggregate_qa(results)
    assert agg["num_unanswerable"] == 2
    assert agg["abstain_accuracy"] == 0.5
    assert "answer_exact" not in agg  # no answerable questions
