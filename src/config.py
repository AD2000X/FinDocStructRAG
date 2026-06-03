"""Centralized paths and model IDs.

Detects Colab vs local and switches the data/output roots automatically, so no
notebook redefines these. The code lives in git (single source of truth); data,
model weights, and outputs live on Google Drive in Colab and under the repo locally.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "FinDocStructRAG"


def in_colab() -> bool:
    """True when running on a Colab VM."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


IN_COLAB = in_colab()

# Colab-only roots.
DRIVE_ROOT = Path("/content/drive/MyDrive") / PROJECT_NAME   # persistent data/outputs
COLAB_REPO_ROOT = Path("/content") / PROJECT_NAME            # git clone on the VM

# Repo root.
#   Local: the project directory (two levels up from this file: src/config.py -> repo).
#   Colab: the git clone under /content.
if IN_COLAB:
    ROOT = COLAB_REPO_ROOT
else:
    ROOT = Path(__file__).resolve().parent.parent

# Data and outputs.
#   Local: kept under the repo (gitignored).
#   Colab: kept on Drive so they survive session loss.
if IN_COLAB:
    DATA_ROOT = DRIVE_ROOT / "data"
    OUTPUT_ROOT = DRIVE_ROOT / "outputs"
else:
    DATA_ROOT = ROOT / "data"
    OUTPUT_ROOT = ROOT / "outputs"

# Output subdirectories (DESIGN_SPEC §12). GT-filled and TATR-predicted are kept
# strictly separate (P4).
TABLES_GT_FILLED = OUTPUT_ROOT / "tables" / "gt_filled"
TABLES_TATR_PREDICTED = OUTPUT_ROOT / "tables" / "tatr_predicted"
TABLES_TATR_RAW = OUTPUT_ROOT / "tables" / "tatr_raw"   # raw TATR debug artifact (not an extraction output)
TABLES_OCR_FILLED = OUTPUT_ROOT / "tables" / "ocr_filled"
TABLES_FAILURES = OUTPUT_ROOT / "tables" / "failures"
RAG_INDEX = OUTPUT_ROOT / "rag_index"
CHUNKS = RAG_INDEX / "chunks"   # serialized table chunks per (text_source, serialization)
QA_DIR = OUTPUT_ROOT / "qa"     # generated + merged QA sets (on Drive)
EVALUATION = OUTPUT_ROOT / "evaluation"
FUNSD_OUTPUT = OUTPUT_ROOT / "funsd"   # Phase 3: relation-linking artifacts (kept separate, P4)
FAILURE_LOGS = OUTPUT_ROOT / "failure_logs"
MANIFESTS = OUTPUT_ROOT / "manifests"
FIGURES = OUTPUT_ROOT / "figures"
LAYOUT_OUTPUT = OUTPUT_ROOT / "layout"   # Phase 2: regions JSON + crop PNGs + manifest

# Hand-authored QA seed (committed in the repo, not on Drive): the manual + unanswerable
# questions that the templated-from-GT generator cannot produce. Eval ground truth, so it
# is version-controlled and travels with git pull.
QA_MANUAL_SEED = ROOT / "qa" / "qa_manual_seed.jsonl"

# Phase 3 FUNSD dataset (raw annotation JSON; gitignored under data/, fetched by
# scripts/fetch_funsd.py). V1 is annotation-only: the JSON carries entity text/bbox/label
# and the GT linking pairs, so no image pixels are loaded. 149 train + 50 test = 199 forms.
FUNSD_ROOT = DATA_ROOT / "raw" / "funsd" / "dataset"
FUNSD_TRAIN = FUNSD_ROOT / "training_data" / "annotations"   # 149 forms
FUNSD_TEST = FUNSD_ROOT / "testing_data" / "annotations"     # 50 forms

# Model IDs (DESIGN_SPEC sections 4.2 and 7; PLAN section 0).
TATR_STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-fin"
TATR_DETECTION_MODEL = "microsoft/table-transformer-detection"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Phase 2 page-level layout detector (DocLayNet classes; id2label[9] == "Table"). Verified on a
# Colab T4 by scripts/smoke_layout_detector.py. NOTE: this checkpoint was saved with transformers
# 4.36.2 and uses a timm resnet50 backbone; transformers>=5's meta-init loader leaves that backbone
# unloaded (degenerate detections), so the Colab env pins transformers==4.49.0 (requirements-colab).
LAYOUT_MODEL = "Aryn/deformable-detr-DocLayNet"

# Answer-generation LLM (single provider, swappable via src/llm_client.py). OpenRouter is an
# OpenAI-compatible gateway; the model id is overridable at runtime by the OPENROUTER_MODEL
# env var, so it can be changed without a code change.
LLM_PROVIDER = "openrouter"
LLM_MODEL = "openai/gpt-4o-mini"
