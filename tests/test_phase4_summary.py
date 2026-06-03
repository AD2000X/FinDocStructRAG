"""Phase 4 summary tests (CPU, synthetic) - Phase 4.

The summarizers take already-loaded metrics dicts (the per-phase evaluation JSONs) or parsed
CSV rows (layout) and return normalized summary dicts; no file IO, no Drive, no gradio is
imported. Fixtures are tiny inline dicts shaped like the real artifacts. Covers each summarizer,
the layout CSV aggregation math (incl. a multi-GT-table page and the table-free FP rate),
missing-artifact tolerance, and the deterministic markdown render (the no-drift property).
"""

from src.phase4_summary import (
    build_summary,
    layout_metrics_from_rows,
    render_metrics_markdown,
    summarize_content,
    summarize_funsd,
    summarize_qa,
    summarize_retrieval,
    summarize_topology,
)


# --- fixtures (shaped like the real outputs/evaluation/*.json) ---

TOPO = {
    "evaluation_type": "topology", "num_samples": 300,
    "row_count_accuracy": 0.79, "col_count_accuracy": 0.987,
    "cell_occupancy_f1": 0.977, "spanning_cell_detection_rate": 0.957,
}

CONTENT = {
    "num_samples": 300,
    "aggregate": {"cell_text_exact_match": 0.804, "numeric_cell_relaxed_match": 0.876,
                  "non_empty_cell_content_f1": 0.977, "alignment_coverage": 0.990},
    "one_to_one": {"cell_text_exact_match": 0.761, "numeric_cell_relaxed_match": 0.825,
                   "non_empty_cell_content_f1": 0.906, "mean_alignment_iou": 0.877},
    "topology_matched_subset": {"num_samples": 234, "metrics": {
        "cell_text_exact_match": 0.819, "numeric_cell_relaxed_match": 0.902,
        "non_empty_cell_content_f1": 0.988}},
}

RETR = {
    "num_questions": 30, "ks": [1, 5, 10], "methods": ["bm25", "dense", "rrf"],
    "corpora": {"gt_markdown": {"bm25": {
        "hit@1": 0.9, "recall@1": 0.9, "mrr@1": 0.9,
        "hit@5": 0.93, "recall@5": 0.93, "mrr@5": 0.91,
        "hit@10": 0.97, "recall@10": 0.97, "mrr@10": 0.92}}},
}

QA = {
    "num_questions": 46, "top_k": 10,
    "configs": {"gt_markdown": {
        "num_questions": 46, "num_answerable": 40,
        "answer_exact": 0.675, "numeric_relaxed": 0.775,
        "citation_hit": 0.8, "abstain_rate": 0.025, "abstain_accuracy": 1.0}},
}

FUNSD = {
    "primary": "test_50.qa_links",
    "results": {
        "test_50": {
            "qa_links": {"precision": 0.946, "recall": 0.590, "f1": 0.727,
                         "tp": 494, "n_pred": 522, "n_gold": 837, "scope": "qa"},
            "all_links": {"precision": 0.946, "recall": 0.464, "f1": 0.623}},
        "train_149": {
            "qa_links": {"precision": 0.919, "recall": 0.521, "f1": 0.665},
            "all_links": {"precision": 0.919, "recall": 0.385, "f1": 0.543}}},
}

# layout CSV rows arrive as strings (csv.DictReader); the aggregator must cast.
POS = [  # GT-table pages
    {"gt_tables": "1", "num_crop_tables": "1", "best_iou_crop": "0.90",
     "matched_50": "1", "matched_75": "1"},
    {"gt_tables": "3", "num_crop_tables": "2", "best_iou_crop": "0.60",
     "matched_50": "2", "matched_75": "1"},
]
NEG = [  # table-free pages
    {"gt_tables": "0", "primary_tables": "0", "fallback_used": "False", "num_crop_tables": "0"},
    {"gt_tables": "0", "primary_tables": "1", "fallback_used": "False", "num_crop_tables": "1"},
    {"gt_tables": "0", "primary_tables": "0", "fallback_used": "True", "num_crop_tables": "0"},
    {"gt_tables": "0", "primary_tables": "0", "fallback_used": "False", "num_crop_tables": "0"},
]
SMOKE = [
    {"crop": "a.png", "valid": "True", "failure_reasons": ""},
    {"crop": "b.png", "valid": "True", "failure_reasons": ""},
    {"crop": "c.png", "valid": "False", "failure_reasons": "rows_not_monotonic"},
    {"crop": "d.png", "valid": "True", "failure_reasons": ""},
]


# --- per-phase summarizers ---


def test_summarize_topology():
    m = summarize_topology(TOPO)
    assert m["n"] == 300
    assert m["row_count_accuracy"] == 0.79
    assert m["spanning_cell_detection_rate"] == 0.957


def test_summarize_content_three_sections():
    m = summarize_content(CONTENT)
    assert m["n"] == 300
    assert m["aggregate"]["cell_text_exact_match"] == 0.804
    assert m["one_to_one"]["mean_alignment_iou"] == 0.877
    assert m["topology_matched_subset"]["n"] == 234
    assert m["topology_matched_subset"]["non_empty_cell_content_f1"] == 0.988


def test_summarize_retrieval_keeps_hit_mrr_drops_recall():
    m = summarize_retrieval(RETR)
    assert m["n"] == 30
    cell = m["corpora"]["gt_markdown"]["bm25"]
    assert set(cell) == {"hit@1", "hit@5", "hit@10", "mrr@10"}
    assert "recall@1" not in cell and "mrr@1" not in cell
    assert cell["mrr@10"] == 0.92


def test_summarize_qa():
    m = summarize_qa(QA)
    assert m["n"] == 46
    cfg = m["configs"]["gt_markdown"]
    assert set(cfg) == {"answer_exact", "numeric_relaxed", "citation_hit", "abstain_rate"}
    assert cfg["answer_exact"] == 0.675


def test_summarize_funsd_headline_from_primary_pointer():
    m = summarize_funsd(FUNSD)
    assert m["primary"] == "test_50.qa_links"
    assert m["headline"] == {"precision": 0.946, "recall": 0.590, "f1": 0.727}
    # per-split results carry only p/r/f1 (no tp/n_pred noise)
    assert set(m["results"]["test_50"]["qa_links"]) == {"precision", "recall", "f1"}
    assert m["results"]["train_149"]["qa_links"]["f1"] == 0.665


# --- layout aggregation (the inline CSV math) ---


def test_layout_metrics_from_rows():
    m = layout_metrics_from_rows(POS, NEG, SMOKE)
    # gt_total=4, crop_total=3, m50=3, m75=2
    assert m["gt_tables"] == 4 and m["crops"] == 3
    assert m["mean_crop_iou"] == 0.75
    assert m["matched@0.50"]["recall"] == 0.75      # 3/4
    assert m["matched@0.50"]["precision"] == 1.0     # 3/3
    assert m["matched@0.75"]["recall"] == 0.5        # 2/4
    assert round(m["matched@0.75"]["precision"], 4) == round(2 / 3, 4)
    # table-free FP: 1 page with a final crop, 1 with a primary detection, out of 4
    assert m["table_free_pages"] == 4
    assert m["crop_fp_rate"] == 0.25
    assert m["primary_fp_rate"] == 0.25
    # crop -> TATR: 3 OK / 1 WARN
    assert m["crop_to_tatr"] == {"n": 4, "ok": 3, "warn": 1, "ok_rate": 0.75}


def test_layout_metrics_empty_is_zero_safe():
    m = layout_metrics_from_rows([], [], [])
    assert m["mean_crop_iou"] == 0.0
    assert m["matched@0.50"]["recall"] == 0.0
    assert m["crop_fp_rate"] == 0.0
    assert m["crop_to_tatr"]["ok_rate"] == 0.0


# --- assembly + render ---


def _full_parts():
    return {
        "topology": summarize_topology(TOPO),
        "content": summarize_content(CONTENT),
        "retrieval": summarize_retrieval(RETR),
        "qa": summarize_qa(QA),
        "layout": layout_metrics_from_rows(POS, NEG, SMOKE),
        "funsd": summarize_funsd(FUNSD),
    }


def test_build_summary_marks_present_and_missing():
    parts = _full_parts()
    parts["layout"] = None                       # simulate a missing artifact
    s = build_summary(parts)
    assert s["layout"] == {"available": False}
    assert s["topology"]["available"] is True
    assert s["funsd"]["headline"]["f1"] == 0.727


def test_render_markdown_deterministic_and_grounded():
    s = build_summary(_full_parts())
    md = render_metrics_markdown(s)
    assert isinstance(md, str) and md.endswith("\n")
    assert md == render_metrics_markdown(s)       # no-drift: pure + deterministic
    assert "0.727" in md                          # FUNSD headline f1 surfaced
    assert "generated by" in md                   # static banner, no timestamp
    assert "recall@" not in md                    # recall@k dropped from the report


def test_render_markdown_tolerates_missing_part():
    parts = _full_parts()
    parts["funsd"] = None
    md = render_metrics_markdown(build_summary(parts))
    assert "not available" in md.lower()
