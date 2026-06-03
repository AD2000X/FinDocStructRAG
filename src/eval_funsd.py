"""FUNSD relation metrics (Phase 3, V1).

Custom set-based precision/recall/F1 over predicted-link vs GT-link sets - no sklearn (P/R/F1
is pure set arithmetic). Split so the metric stays pure (no predictor inside) and the
form-runner is separate:

- prf1: one (pred, gold) set pair -> P/R/F1.
- evaluate_pairs: micro P/R/F1 over many prebuilt (pred, gold) pairs (trivially unit-tested).
- evaluate_forms: runs the predictor over forms for a scope, then delegates to evaluate_pairs.

Scopes (see docs/phase3_brief.md):
- "qa": predicted directed (question_id, answer_id) vs qa_gold_links (primary).
- "all": the same QA predictions cast to undirected frozensets vs all_gold_links - a coverage
  diagnostic ("how many of ALL GT links does the QA-only heuristic recover"), not a second
  predictor.
"""

from __future__ import annotations

from src.funsd_extraction import (
    HeuristicParams,
    FunsdForm,
    all_gold_links,
    predict_qa_links,
    qa_gold_links,
)


def _prf(tp: int, n_pred: int, n_gold: int) -> dict:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "n_pred": n_pred, "n_gold": n_gold,
            "precision": precision, "recall": recall, "f1": f1}


def prf1(pred: set, gold: set) -> dict:
    """P/R/F1 for a single (pred, gold) set pair. Empty pred or gold scores 0 on that side."""
    return _prf(len(pred & gold), len(pred), len(gold))


def evaluate_pairs(per_form: list[tuple[set, set]]) -> dict:
    """Micro P/R/F1 over prebuilt (pred, gold) pairs: sum tp/|pred|/|gold| across forms, then
    compute once. Pure - no predictor or params."""
    tp = sum(len(p & g) for p, g in per_form)
    n_pred = sum(len(p) for p, _ in per_form)
    n_gold = sum(len(g) for _, g in per_form)
    out = _prf(tp, n_pred, n_gold)
    out["num_forms"] = len(per_form)
    return out


def evaluate_forms(forms: list[FunsdForm], scope: str,
                   params: HeuristicParams = HeuristicParams()) -> dict:
    """Run the heuristic over forms and score it for a scope ("qa" or "all")."""
    if scope not in ("qa", "all"):
        raise ValueError(f"unknown scope: {scope!r} (use 'qa' or 'all')")

    per_form: list[tuple[set, set]] = []
    for form in forms:
        pred = predict_qa_links(form, params)
        if scope == "qa":
            per_form.append((pred, qa_gold_links(form)))
        else:  # "all": score QA predictions as undirected pairs against every GT link
            per_form.append(({frozenset(pair) for pair in pred}, all_gold_links(form)))

    out = evaluate_pairs(per_form)
    out["scope"] = scope
    return out
