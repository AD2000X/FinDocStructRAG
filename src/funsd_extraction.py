"""FUNSD relation-linking baseline (Phase 3, V1).

Annotation-only and CPU-only: the FUNSD annotation JSON already carries each entity's text,
bbox, label ({question, answer, header, other}), and the GT `linking` pairs, so nothing here
loads image pixels. The module provides the data contract (parse + normalize + dedupe GT
links), the two link scopes used for reporting, and a deterministic spatial predictor.

Predictor: per-answer argmax + distance gate. Each `answer` scores every `question`
candidate (a same-row right-side relation or a below relation), the candidate set is filtered
by a distance gate, and the answer is linked to its single best question if that score clears
a floor. Distances are normalized by the form's median entity height so one set of thresholds
works across differently-scaled scans.

GT links are stored undirected (FUNSD records a link on both endpoints); `qa_gold_links`
canonicalizes question+answer pairs to a directed (question_id, answer_id), `all_gold_links`
keeps the full undirected set. See docs/phase3_brief.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TypedDict

QUESTION = "question"
ANSWER = "answer"


class FunsdEntity(TypedDict):
    """A FUNSD form entity. box is [x0, y0, x1, y1] in image pixels (top-left origin)."""

    id: int
    label: str
    text: str
    box: list[float]


class FunsdForm(TypedDict):
    """A parsed form. gold_links is the deduped undirected GT link set (cast to list only
    when serializing to JSON)."""

    form_id: str
    entities: list[FunsdEntity]
    gold_links: set[frozenset[int]]


@dataclass(frozen=True)
class HeuristicParams:
    """Tunable surface for the relation heuristic. Defaults are a-priori; any fitting is done
    on the FUNSD train split only (never on the reported test set). Distances are in units of
    the form's median entity height.

    Two clearly-separated knobs, not one fuzzy gate:
    - max_distance_units: the distance gate that filters candidates too far to be plausible.
    - min_score: the floor the per-answer argmax winner must clear to be emitted.
    """

    right_base: float = 1.0          # base score for a same-row right-side answer
    below_base: float = 0.7          # base score for a below answer
    right_band_tol: float = 0.7      # vertical-center tolerance for "same row" (H units)
    below_align_tol: float = 1.0     # left-edge tolerance for "below" alignment (H units)
    align_boost: float = 0.5         # reward for tighter band / left-edge alignment
    dist_penalty: float = 0.3        # score lost per median-height of gap
    max_distance_units: float = 8.0  # distance gate: reject candidates beyond this (H units)
    min_score: float = 0.0           # score floor on the chosen link


# --- parsing ---


def parse_funsd_form(data: dict, form_id: str) -> FunsdForm:
    """Build a FunsdForm from the FUNSD JSON shape ({"form": [ {id, label, box, ...}, ... ]}).

    Links are collected across all entities and deduped to undirected frozensets; self-links
    and links referencing a missing id are dropped.
    """
    raw = data.get("form", [])
    entities: list[FunsdEntity] = []
    ids: set[int] = set()
    for e in raw:
        eid = int(e["id"])
        entities.append(FunsdEntity(
            id=eid,
            label=str(e.get("label", "")),
            text=str(e.get("text", "")),
            box=[float(v) for v in e["box"]],
        ))
        ids.add(eid)

    gold: set[frozenset[int]] = set()
    for e in raw:
        for pair in e.get("linking", []):
            a, b = int(pair[0]), int(pair[1])
            if a == b or a not in ids or b not in ids:
                continue
            gold.add(frozenset((a, b)))

    return FunsdForm(form_id=form_id, entities=entities, gold_links=gold)


def parse_funsd_json(path) -> FunsdForm:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return parse_funsd_form(data, path.stem)


def load_funsd_split(annotations_dir) -> list[FunsdForm]:
    """Parse every *.json in a FUNSD annotations directory (sorted for determinism)."""
    return [parse_funsd_json(p) for p in sorted(Path(annotations_dir).glob("*.json"))]


# --- GT link scopes ---


def _labels(form: FunsdForm) -> dict[int, str]:
    return {e["id"]: e["label"] for e in form["entities"]}


def qa_gold_links(form: FunsdForm) -> set[tuple[int, int]]:
    """Question+answer GT links as directed (question_id, answer_id) pairs (primary scope)."""
    label = _labels(form)
    out: set[tuple[int, int]] = set()
    for pair in form["gold_links"]:
        a, b = tuple(pair)
        la, lb = label[a], label[b]
        if {la, lb} == {QUESTION, ANSWER}:
            q, ans = (a, b) if la == QUESTION else (b, a)
            out.add((q, ans))
    return out


def all_gold_links(form: FunsdForm) -> set[frozenset[int]]:
    """The full deduped undirected GT link set (secondary coverage scope)."""
    return set(form["gold_links"])


# --- heuristic predictor ---


def median_entity_height(entities: list[FunsdEntity]) -> float:
    heights = [e["box"][3] - e["box"][1] for e in entities]
    return float(median(heights)) if heights else 1.0


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _score(q: FunsdEntity, a: FunsdEntity, h: float, params: HeuristicParams) -> float | None:
    """Best score for linking answer a to question q, or None if a is not a valid candidate.

    Right-side and below relations compete in the same max (not a strict fallback): whichever
    geometric relation gives the higher score wins, after the distance gate.
    """
    qx0, qy0, qx1, qy1 = q["box"]
    ax0, ay0, ax1, ay1 = a["box"]
    qcx, qcy = (qx0 + qx1) / 2, (qy0 + qy1) / 2
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    h = h if h > 1e-6 else 1e-6

    candidates: list[tuple[float, float]] = []  # (score, distance_units)

    # same-row right-side: A vertically aligned with Q and to its right
    band = abs(acy - qcy) / h
    if acx > qcx and band <= params.right_band_tol:
        hgap = max(0.0, ax0 - qx1) / h
        score = (params.right_base
                 + params.align_boost * (params.right_band_tol - band)
                 - params.dist_penalty * hgap)
        candidates.append((score, hgap))

    # below: A under Q, horizontally overlapping or left-aligned
    left_off = abs(ax0 - qx0) / h
    if acy > qcy and (_overlap_1d(qx0, qx1, ax0, ax1) > 0 or left_off <= params.below_align_tol):
        vgap = max(0.0, ay0 - qy1) / h
        score = (params.below_base
                 + params.align_boost * max(0.0, params.below_align_tol - left_off)
                 - params.dist_penalty * vgap)
        candidates.append((score, vgap))

    valid = [s for s, d in candidates if d <= params.max_distance_units]
    return max(valid) if valid else None


def predict_qa_links(form: FunsdForm,
                     params: HeuristicParams = HeuristicParams()) -> set[tuple[int, int]]:
    """Per-answer argmax: each answer links to its single best-scoring question above the gate
    and the score floor. Returns directed (question_id, answer_id) pairs."""
    entities = form["entities"]
    questions = [e for e in entities if e["label"] == QUESTION]
    answers = [e for e in entities if e["label"] == ANSWER]
    h = median_entity_height(entities)

    links: set[tuple[int, int]] = set()
    for a in answers:
        best_q: FunsdEntity | None = None
        best_score: float | None = None
        for q in questions:
            s = _score(q, a, h, params)
            if s is None:
                continue
            # deterministic: higher score wins, ties break to the lower question id
            if (best_score is None or s > best_score
                    or (s == best_score and q["id"] < best_q["id"])):
                best_score, best_q = s, q
        if best_q is not None and best_score >= params.min_score:
            links.add((best_q["id"], a["id"]))
    return links
