"""Phase 4 capstone: aggregate the per-phase evaluation artifacts into one summary.

Pure helpers only - no file IO, no Drive, no gradio. Each summarizer takes an already-loaded
metrics dict (the per-phase evaluation JSON) or parsed CSV rows (layout) and returns a normalized
summary dict; scripts/build_phase4_summary.py does the reading/writing. `build_summary` assembles
the parts (a missing one becomes `{"available": False}`), and `render_metrics_markdown` turns the
summary into the deterministic, paste-ready table committed at reports/phase4_metrics.md - the
report prose reads those numbers, they are never hand-copied. See docs/phase4_brief.md.
"""

from __future__ import annotations

from statistics import mean

# --- per-phase summarizers (input: the loaded metrics JSON dict) ---

_CONTENT_KEYS = ["cell_text_exact_match", "numeric_cell_relaxed_match", "non_empty_cell_content_f1"]
_RETRIEVAL_KEYS = ["hit@1", "hit@5", "hit@10", "mrr@10"]  # recall@k == hit@k here, so dropped
_QA_KEYS = ["answer_exact", "numeric_relaxed", "citation_hit", "abstain_rate"]
_PRF = ("precision", "recall", "f1")

PHASES = ["topology", "content", "retrieval", "qa", "layout", "funsd"]


def summarize_topology(d: dict) -> dict:
    return {
        "n": d["num_samples"],
        "row_count_accuracy": d["row_count_accuracy"],
        "col_count_accuracy": d["col_count_accuracy"],
        "cell_occupancy_f1": d["cell_occupancy_f1"],
        "spanning_cell_detection_rate": d["spanning_cell_detection_rate"],
    }


def summarize_content(d: dict) -> dict:
    agg, o2o = d["aggregate"], d["one_to_one"]
    sub = d["topology_matched_subset"]["metrics"]
    pick = lambda m: {k: m[k] for k in _CONTENT_KEYS}
    return {
        "n": d["num_samples"],
        "aggregate": {**pick(agg), "alignment_coverage": agg["alignment_coverage"]},
        "one_to_one": {**pick(o2o), "mean_alignment_iou": o2o["mean_alignment_iou"]},
        "topology_matched_subset": {"n": d["topology_matched_subset"]["num_samples"], **pick(sub)},
    }


def summarize_retrieval(d: dict) -> dict:
    corpora = {
        corpus: {method: {k: m[k] for k in _RETRIEVAL_KEYS} for method, m in methods.items()}
        for corpus, methods in d["corpora"].items()
    }
    return {"n": d["num_questions"], "methods": d["methods"], "corpora": corpora}


def summarize_qa(d: dict) -> dict:
    configs = {name: {k: m[k] for k in _QA_KEYS} for name, m in d["configs"].items()}
    return {"n": d["num_questions"], "configs": configs}


def summarize_funsd(d: dict) -> dict:
    split, scope_key = d["primary"].split(".", 1)   # "test_50.qa_links" -> "test_50", "qa_links"
    head = d["results"][split][scope_key]
    results = {
        sp: {sk: {k: sv[k] for k in _PRF} for sk, sv in scopes.items()}
        for sp, scopes in d["results"].items()
    }
    return {"primary": d["primary"], "headline": {k: head[k] for k in _PRF}, "results": results}


# --- layout aggregation (Phase 2 has no JSON; aggregate the staged CSV rows inline) ---


def _i(v) -> int:
    return int(v)


def _f(v) -> float:
    return float(v)


def _truthy(v) -> bool:
    return str(v).strip().lower() == "true"


def layout_metrics_from_rows(pos_rows: list[dict], neg_rows: list[dict],
                             smoke_rows: list[dict]) -> dict:
    """Aggregate diagnostic_pos.csv (GT-table pages), diagnostic_neg.csv (table-free pages), and
    smoke_structure.csv (crop -> TATR) rows. Rows are dicts of strings (csv.DictReader); cast here.
    Mirrors the table-level matching + FP definitions printed by scripts/eval_layout_iou.py.
    """
    gt_total = sum(_i(r["gt_tables"]) for r in pos_rows)
    crop_total = sum(_i(r["num_crop_tables"]) for r in pos_rows)
    m50 = sum(_i(r["matched_50"]) for r in pos_rows)
    m75 = sum(_i(r["matched_75"]) for r in pos_rows)

    def matched(m: int) -> dict:
        return {"recall": m / gt_total if gt_total else 0.0,
                "precision": m / crop_total if crop_total else 0.0}

    n_neg = len(neg_rows)
    primary_fp = sum(1 for r in neg_rows if _i(r["primary_tables"]) > 0)
    crop_fp = sum(1 for r in neg_rows if _i(r["num_crop_tables"]) > 0)

    n_smoke = len(smoke_rows)
    ok = sum(1 for r in smoke_rows if _truthy(r["valid"]))

    return {
        "n_gt_pages": len(pos_rows),
        "gt_tables": gt_total,
        "crops": crop_total,
        "mean_crop_iou": mean(_f(r["best_iou_crop"]) for r in pos_rows) if pos_rows else 0.0,
        "matched@0.50": matched(m50),
        "matched@0.75": matched(m75),
        "table_free_pages": n_neg,
        "primary_fp_rate": primary_fp / n_neg if n_neg else 0.0,
        "crop_fp_rate": crop_fp / n_neg if n_neg else 0.0,
        "crop_to_tatr": {"n": n_smoke, "ok": ok, "warn": n_smoke - ok,
                         "ok_rate": ok / n_smoke if n_smoke else 0.0},
    }


# --- assembly + render ---


def build_summary(parts: dict) -> dict:
    """Assemble per-phase summary dicts. parts maps a phase name (see PHASES) to its summary dict
    or None; present parts get `available: True`, a missing one becomes `{"available": False}`."""
    out: dict = {}
    for name in PHASES:
        val = parts.get(name)
        out[name] = {"available": True, **val} if val is not None else {"available": False}
    return out


_BANNER = "<!-- generated by scripts/build_phase4_summary.py - do not edit by hand -->"


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def render_metrics_markdown(summary: dict) -> str:
    """Deterministic markdown for reports/phase4_metrics.md. Pure: same summary -> same bytes
    (the no-drift property). A missing phase renders as '_Not available._'."""
    L: list[str] = [_BANNER, "", "# Phase 4 metrics summary", "",
                    "Generated from `outputs/evaluation/phase4_summary.json`; do not edit by hand.",
                    ""]

    # Table extraction
    L.append("## Table extraction (Phase 1A topology, Phase 1B content)")
    t = summary["topology"]
    if t.get("available"):
        L += _table(["topology metric", f"value (n={t['n']})"], [
            ["row count accuracy", _fmt(t["row_count_accuracy"])],
            ["col count accuracy", _fmt(t["col_count_accuracy"])],
            ["cell occupancy F1", _fmt(t["cell_occupancy_f1"])],
            ["spanning cell detection", _fmt(t["spanning_cell_detection_rate"])],
        ])
    else:
        L.append("_Not available._")
    L.append("")
    c = summary["content"]
    if c.get("available"):
        L += _table(["content (cell-level)", "exact", "numeric", "non-empty F1"], [
            [f"aggregate (n={c['n']})", _fmt(c["aggregate"]["cell_text_exact_match"]),
             _fmt(c["aggregate"]["numeric_cell_relaxed_match"]),
             _fmt(c["aggregate"]["non_empty_cell_content_f1"])],
            ["one-to-one (IoU>=0.5)", _fmt(c["one_to_one"]["cell_text_exact_match"]),
             _fmt(c["one_to_one"]["numeric_cell_relaxed_match"]),
             _fmt(c["one_to_one"]["non_empty_cell_content_f1"])],
            [f"topology-matched (n={c['topology_matched_subset']['n']})",
             _fmt(c["topology_matched_subset"]["cell_text_exact_match"]),
             _fmt(c["topology_matched_subset"]["numeric_cell_relaxed_match"]),
             _fmt(c["topology_matched_subset"]["non_empty_cell_content_f1"])],
        ])
        L.append("")
        L.append(f"mean alignment IoU (one-to-one): {_fmt(c['one_to_one']['mean_alignment_iou'])}")
    else:
        L.append("_Not available._")
    L.append("")

    # Layout
    L.append("## Layout (Phase 2 DocLayNet crop)")
    g = summary["layout"]
    if g.get("available"):
        cr = g["crop_to_tatr"]
        L += _table(["layout metric", "value"], [
            ["mean crop IoU (GT-table pages)", _fmt(g["mean_crop_iou"])],
            ["matched@0.50 (recall / precision)",
             f"{_fmt(g['matched@0.50']['recall'])} / {_fmt(g['matched@0.50']['precision'])}"],
            ["matched@0.75 (recall / precision)",
             f"{_fmt(g['matched@0.75']['recall'])} / {_fmt(g['matched@0.75']['precision'])}"],
            ["table-free crop FP rate", _fmt(g["crop_fp_rate"])],
            ["crop -> TATR OK rate", f"{_fmt(cr['ok_rate'])} ({cr['ok']}/{cr['n']})"],
        ])
    else:
        L.append("_Not available._")
    L.append("")

    # Retrieval
    L.append("## Retrieval (Phase 1C, table chunks)")
    r = summary["retrieval"]
    if r.get("available"):
        rows = []
        for corpus in sorted(r["corpora"]):
            for method in sorted(r["corpora"][corpus]):
                m = r["corpora"][corpus][method]
                rows.append([corpus, method, _fmt(m["hit@1"]), _fmt(m["hit@5"]),
                             _fmt(m["hit@10"]), _fmt(m["mrr@10"])])
        L += _table([f"corpus (n={r['n']})", "method", "hit@1", "hit@5", "hit@10", "MRR@10"], rows)
    else:
        L.append("_Not available._")
    L.append("")

    # QA
    L.append("## Table QA (Phase 1C, answer generation)")
    q = summary["qa"]
    if q.get("available"):
        rows = [[cfg, _fmt(m["answer_exact"]), _fmt(m["numeric_relaxed"]),
                 _fmt(m["citation_hit"]), _fmt(m["abstain_rate"])]
                for cfg, m in sorted(q["configs"].items())]
        L += _table([f"config (n={q['n']})", "answer exact", "numeric relaxed",
                     "citation hit", "abstain rate"], rows)
    else:
        L.append("_Not available._")
    L.append("")

    # FUNSD
    L.append("## FUNSD relations (Phase 3)")
    f = summary["funsd"]
    if f.get("available"):
        h = f["headline"]
        L.append(f"headline ({f['primary']}): "
                 f"P {_fmt(h['precision'])} / R {_fmt(h['recall'])} / F1 {_fmt(h['f1'])}")
        L.append("")
        rows = []
        for split in sorted(f["results"]):
            for scope in sorted(f["results"][split]):
                v = f["results"][split][scope]
                rows.append([split, scope, _fmt(v["precision"]), _fmt(v["recall"]), _fmt(v["f1"])])
        L += _table(["split", "scope", "precision", "recall", "f1"], rows)
    else:
        L.append("_Not available._")
    L.append("")

    return "\n".join(L) + "\n"
