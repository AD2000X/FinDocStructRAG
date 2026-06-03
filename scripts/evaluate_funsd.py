"""Phase 3: FUNSD relation-linking evaluation (CPU, no GPU, no network).

Annotation-only baseline. Reads the local FUNSD annotation JSON, runs the deterministic
per-answer-argmax heuristic, and reports relation P/R/F1 across split x scope:

    python scripts/evaluate_funsd.py

Splits:  train_149 (tuning/dev), test_50 (PRIMARY headline), all_199 = train + test
         (secondary, NOT held-out - it contains the 50 test + 149 tuned forms), debug_20.
Scopes:  qa_links (primary, question->answer) and all_links (secondary coverage diagnostic).

Headline metric: test_50.qa_links.micro_f1. Writes
outputs/evaluation/phase3_funsd_relations.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.eval_funsd import evaluate_forms  # noqa: E402
from src.funsd_extraction import HeuristicParams, load_funsd_split  # noqa: E402

SCOPES = ("qa", "all")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--debug-n", type=int, default=20,
                    help="size of the debug split (first N train forms; parser/CLI smoke only)")
    args = ap.parse_args()

    if not config.FUNSD_TRAIN.is_dir() or not config.FUNSD_TEST.is_dir():
        raise SystemExit(
            f"FUNSD annotations not found under {config.FUNSD_ROOT}\n"
            f"  expected: {config.FUNSD_TRAIN} and {config.FUNSD_TEST}\n"
            f"  run: python scripts/fetch_funsd.py")

    train = load_funsd_split(config.FUNSD_TRAIN)
    test = load_funsd_split(config.FUNSD_TEST)
    debug_n = min(args.debug_n, len(train))
    splits = {
        "train_149": train,
        "test_50": test,
        "all_199": train + test,
        f"debug_{debug_n}": train[:debug_n],
    }

    params = HeuristicParams()
    report: dict = {
        "params": params.__dict__,
        "split_sizes": {name: len(forms) for name, forms in splits.items()},
        "primary": "test_50.qa_links",
        "note": ("all_199 contains the 50 test + 149 tuned forms and is NOT held-out; "
                 "the held-out headline is test_50.qa_links. all_links is a coverage "
                 "diagnostic of the QA-only predictor, not a second predictor."),
        "results": {},
    }
    for name, forms in splits.items():
        report["results"][name] = {
            f"{scope}_links": evaluate_forms(forms, scope, params) for scope in SCOPES
        }

    out_dir = config.EVALUATION
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase3_funsd_relations.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary: headline first, then secondaries.
    def row(split: str, scope: str) -> str:
        m = report["results"][split][f"{scope}_links"]
        return (f"{split:<10} {scope+'_links':<10} "
                f"P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
                f"(tp {m['tp']} / pred {m['n_pred']} / gold {m['n_gold']}, n={m['num_forms']})")

    print("HEADLINE (held-out):")
    print("  " + row("test_50", "qa"))
    print("\nSecondary:")
    print("  " + row("all_199", "qa"))
    print("  " + row("test_50", "all"))
    print("  " + row("all_199", "all"))
    print("  " + row("train_149", "qa") + "   [dev/tuning split]")
    print(f"\nreport -> {out_path}")


if __name__ == "__main__":
    main()
