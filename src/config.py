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
EVALUATION = OUTPUT_ROOT / "evaluation"
FAILURE_LOGS = OUTPUT_ROOT / "failure_logs"
MANIFESTS = OUTPUT_ROOT / "manifests"
FIGURES = OUTPUT_ROOT / "figures"

# Model IDs (DESIGN_SPEC §4.2, §7; PLAN §0).
TATR_STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-fin"
TATR_DETECTION_MODEL = "microsoft/table-transformer-detection"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Answer-generation LLM (single provider, swappable via src/llm_client.py).
LLM_PROVIDER = "gemini"
