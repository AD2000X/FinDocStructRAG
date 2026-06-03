"""FUNSD relation-linking tests (CPU, synthetic) - Phase 3.

Annotation-only baseline: forms are built from inline dicts in the same shape FUNSD's
annotation JSON uses ({"form": [ {id, label, box, text, linking}, ... ]}); no raw dataset
and no image pixels are touched. Covers the parser, link dedupe/scoping, the per-answer
argmax + distance-gate heuristic, and the set-based P/R/F1 metric.
"""

import json

from dataclasses import replace

from src.funsd_extraction import (
    HeuristicParams,
    all_gold_links,
    median_entity_height,
    parse_funsd_form,
    parse_funsd_json,
    predict_qa_links,
    qa_gold_links,
)
from src.eval_funsd import evaluate_forms, evaluate_pairs, prf1


def _ent(eid, label, box, *, text="", linking=None):
    return {"id": eid, "label": label, "box": list(box), "text": text,
            "linking": list(linking or [])}


def _form(entities, form_id="f0"):
    return parse_funsd_form({"form": entities}, form_id)


# --- parsing ---


def test_parse_form_entities_and_links():
    form = _form([
        _ent(0, "question", [0, 0, 50, 20], text="Name", linking=[[0, 1]]),
        _ent(1, "answer", [60, 0, 120, 20], text="Bob", linking=[[0, 1]]),
    ])
    assert form["form_id"] == "f0"
    assert len(form["entities"]) == 2
    assert form["entities"][0]["label"] == "question"
    assert form["entities"][0]["box"] == [0.0, 0.0, 50.0, 20.0]
    assert form["gold_links"] == {frozenset((0, 1))}


def test_parse_json_reads_file(tmp_path):
    p = tmp_path / "form.json"
    p.write_text(json.dumps({"form": [
        _ent(0, "question", [0, 0, 1, 1], linking=[[0, 1]]),
        _ent(1, "answer", [2, 0, 3, 1], linking=[[0, 1]]),
    ]}), encoding="utf-8")
    form = parse_funsd_json(p)
    assert form["form_id"] == "form"
    assert form["gold_links"] == {frozenset((0, 1))}


def test_links_dedupe_bidirectional_and_drop_invalid():
    # link recorded on both ends, plus a self-link and a dangling id -> one clean pair
    form = _form([
        _ent(0, "question", [0, 0, 50, 20], linking=[[0, 1], [0, 0]]),
        _ent(1, "answer", [60, 0, 120, 20], linking=[[1, 0], [1, 99]]),
    ])
    assert form["gold_links"] == {frozenset((0, 1))}


# --- link scopes ---


def test_qa_link_filter_keeps_only_question_answer():
    form = _form([
        _ent(0, "question", [0, 0, 50, 20], linking=[[0, 1], [0, 2]]),
        _ent(1, "answer", [60, 0, 120, 20], linking=[[0, 1]]),
        _ent(2, "question", [0, 30, 50, 50], linking=[[0, 2]]),  # question-question
        _ent(3, "header", [0, 60, 50, 80], linking=[[3, 0]]),    # header-question
    ])
    assert qa_gold_links(form) == {(0, 1)}            # directed question -> answer
    assert all_gold_links(form) == {
        frozenset((0, 1)), frozenset((0, 2)), frozenset((0, 3))}


def test_qa_link_canonicalizes_direction_regardless_of_record_order():
    # link stored answer-first; qa scope still emits (question_id, answer_id)
    form = _form([
        _ent(0, "answer", [60, 0, 120, 20], linking=[[0, 1]]),
        _ent(1, "question", [0, 0, 50, 20], linking=[[0, 1]]),
    ])
    assert qa_gold_links(form) == {(1, 0)}


def test_header_question_link_excluded_from_qa_scope():
    form = _form([
        _ent(0, "header", [0, 0, 50, 20], linking=[[0, 1]]),
        _ent(1, "question", [0, 30, 50, 50], linking=[[0, 1]]),
    ])
    assert qa_gold_links(form) == set()
    assert all_gold_links(form) == {frozenset((0, 1))}


# --- metrics ---


def test_prf1_basic():
    m = prf1({(1, 2), (3, 4)}, {(1, 2), (5, 6)})
    assert m["tp"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


def test_prf1_empty_pred_is_zero():
    m = prf1(set(), {(1, 2)})
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_evaluate_pairs_micro_aggregation():
    per_form = [
        ({(1, 2)}, {(1, 2), (3, 4)}),   # tp 1, pred 1, gold 2
        (set(), {(5, 6)}),              # tp 0, pred 0, gold 1
        ({(7, 8)}, {(7, 8)}),           # tp 1, pred 1, gold 1
    ]
    m = evaluate_pairs(per_form)
    assert m["num_forms"] == 3
    assert m["tp"] == 2 and m["n_pred"] == 2 and m["n_gold"] == 4
    assert m["precision"] == 1.0     # 2 / 2
    assert m["recall"] == 0.5        # 2 / 4
    assert round(m["f1"], 4) == round(2 * 1.0 * 0.5 / 1.5, 4)


# --- heuristic predictor ---


def test_same_row_right_side_answer_is_linked():
    form = _form([
        _ent(0, "question", [0, 0, 50, 20]),
        _ent(1, "answer", [60, 0, 120, 20]),
    ])
    assert predict_qa_links(form) == {(0, 1)}


def test_below_candidate_can_win_against_a_valid_right_candidate():
    # answer A has both a valid same-row question (far left) and a question directly above;
    # the closer "below" question wins the per-answer argmax (not a no-right-side fallback)
    form = _form([
        _ent(0, "question", [0, 100, 40, 120]),      # same row, far to A's left
        _ent(1, "question", [100, 70, 150, 90]),     # directly above A, close
        _ent(2, "answer", [100, 100, 150, 120]),
    ])
    assert predict_qa_links(form) == {(1, 2)}


def test_other_label_is_never_linked():
    form = _form([
        _ent(0, "question", [0, 0, 50, 20]),
        _ent(1, "other", [60, 0, 120, 20]),      # sits where an answer would, but is "other"
        _ent(2, "header", [0, 30, 50, 50]),
    ])
    assert predict_qa_links(form) == set()


def test_per_answer_argmax_links_only_the_nearer_question():
    form = _form([
        _ent(0, "question", [0, 0, 40, 20]),     # far left
        _ent(1, "question", [60, 0, 90, 20]),    # nearer to A
        _ent(2, "answer", [100, 0, 150, 20]),
    ])
    assert predict_qa_links(form) == {(1, 2)}


def test_distance_gate_drops_far_answer():
    # right-side candidate exists and scores above min_score, but its normalized distance
    # exceeds max_distance_units -> the gate (not the score floor) drops it
    form = _form([
        _ent(0, "question", [0, 0, 50, 20]),
        _ent(1, "answer", [110, 0, 160, 20]),    # hgap = (110-50)/20 = 3.0 median-heights
    ])
    tight = replace(HeuristicParams(), max_distance_units=2.0)
    assert predict_qa_links(form, tight) == set()
    assert predict_qa_links(form) == {(0, 1)}    # default gate (8.0) keeps it


def test_median_entity_height():
    form = _form([
        _ent(0, "question", [0, 0, 10, 10]),     # h 10
        _ent(1, "answer", [0, 0, 10, 30]),       # h 30
        _ent(2, "answer", [0, 0, 10, 20]),       # h 20
    ])
    assert median_entity_height(form["entities"]) == 20.0


# --- form-level evaluation (predictor + scope, end to end on a synthetic form) ---


def test_evaluate_forms_qa_scope_perfect_form():
    forms = [_form([
        _ent(0, "question", [0, 0, 50, 20], linking=[[0, 1]]),
        _ent(1, "answer", [60, 0, 120, 20], linking=[[0, 1]]),
    ])]
    m = evaluate_forms(forms, "qa")
    assert m["scope"] == "qa"
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_evaluate_forms_all_scope_counts_uncovered_links():
    # one QA link (predictable) + one header-question link the QA-only predictor cannot cover
    forms = [_form([
        _ent(0, "question", [0, 0, 50, 20], linking=[[0, 1], [2, 0]]),
        _ent(1, "answer", [60, 0, 120, 20], linking=[[0, 1]]),
        _ent(2, "header", [0, 30, 50, 50], linking=[[2, 0]]),
    ])]
    m = evaluate_forms(forms, "all")
    assert m["scope"] == "all"
    assert m["n_gold"] == 2 and m["tp"] == 1     # only the QA link is recovered
    assert m["recall"] == 0.5
