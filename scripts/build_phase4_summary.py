#!/usr/bin/env python3
"""Build the Phase 4 capstone summary from the per-phase evaluation artifacts.

Reads the five metrics JSONs + the three Phase 2 layout CSVs from outputs/, aggregates them with
the pure helpers in src/phase4_summary.py, and writes:
  - outputs/evaluation/phase4_summary.json   (gitignored machine artifact)
  - reports/phase4_metrics.md                (committed, paste-ready; the report reads these)
A missing artifact degrades gracefully (its section is marked unavailable). Layout has no JSON, so
it is aggregated inline from diagnostic_pos.csv / diagnostic_neg.csv / smoke_structure.csv. The
markdown is written with LF newlines so the no-drift gate holds across Windows and Colab. See
docs/phase4_brief.md.

Usage:
    python scripts/build_phase4_summary.py [--run-id mvp_rand]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json

from src import config
from src import phase4_summary as p4


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_csv(path: Path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _layout_part(layout_dir: Path):
    """Aggregate the three staged layout CSVs; None unless all are present."""
    pos = _load_csv(layout_dir / "diagnostic_pos.csv")
    neg = _load_csv(layout_dir / "diagnostic_neg.csv")
    smoke = _load_csv(layout_dir / "smoke_structure.csv")
    if pos is None or neg is None or smoke is None:
        return None
    return p4.layout_metrics_from_rows(pos, neg, smoke)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Phase 4 capstone summary.")
    ap.add_argument("--run-id", default="mvp_rand",
                    help="run-id suffix of the Phase 1A/1B deliverable artifacts")
    args = ap.parse_args()

    ev = config.EVALUATION
    rag = ev / "rag"
    topo = _load_json(ev / f"phase1a_topology_{args.run_id}.json")
    content = _load_json(ev / f"phase1b_content_{args.run_id}.json")
    retr = _load_json(rag / "phase1c_retrieval.json")
    qa = _load_json(rag / "phase1c_qa.json")
    funsd = _load_json(ev / "phase3_funsd_relations.json")

    parts = {
        "topology": p4.summarize_topology(topo) if topo else None,
        "content": p4.summarize_content(content) if content else None,
        "retrieval": p4.summarize_retrieval(retr) if retr else None,
        "qa": p4.summarize_qa(qa) if qa else None,
        "layout": _layout_part(config.LAYOUT_OUTPUT),
        "funsd": p4.summarize_funsd(funsd) if funsd else None,
    }
    summary = p4.build_summary(parts)

    summary_path = ev / "phase4_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = config.ROOT / "reports" / "phase4_metrics.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8", newline="") as f:   # newline="": LF verbatim
        f.write(p4.render_metrics_markdown(summary))

    print("Phase 4 summary - artifact availability:")
    for name in p4.PHASES:
        print(f"  {name:<10} {'OK' if summary[name].get('available') else 'MISSING'}")
    if summary["funsd"].get("available"):
        h = summary["funsd"]["headline"]
        print(f"\nFUNSD headline ({summary['funsd']['primary']}): "
              f"P {h['precision']:.3f} / R {h['recall']:.3f} / F1 {h['f1']:.3f}")
    print(f"\nsummary -> {summary_path}")
    print(f"metrics  -> {md_path}")


if __name__ == "__main__":
    main()
