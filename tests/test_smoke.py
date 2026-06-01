"""Phase 0 smoke tests (local CPU).

These must actually run and pass: they verify the package imports, config path
detection works, the failure logger initializes, and the canonical schema is the
expected type. Heavy GPU-dependent tests arrive in later phases.
"""

from pathlib import Path


def test_import_src():
    """from src import ... does not error (no GPU deps pulled in)."""
    from src import config, canonical_schema, failure_logger, tatr_postprocess

    assert config is not None
    assert canonical_schema is not None
    assert failure_logger is not None
    assert tatr_postprocess is not None


def test_config_paths():
    """Config path detection returns valid Path values and locked model IDs."""
    from src import config

    assert config.PROJECT_NAME == "FinDocStructRAG"
    assert isinstance(config.ROOT, Path)
    assert isinstance(config.DATA_ROOT, Path)
    assert isinstance(config.OUTPUT_ROOT, Path)
    assert isinstance(config.IN_COLAB, bool)
    # Locked decisions (PLAN §0).
    assert config.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert config.LLM_PROVIDER == "openrouter"


def test_failure_logger_init(tmp_path):
    """FailureLogger initializes and records a failure to JSONL."""
    from src.failure_logger import FailureLogger

    logger = FailureLogger(tmp_path / "failures.jsonl")
    assert len(logger) == 0

    logger.log(sample_id="s1", phase="phase0", error_type="unknown", message="test")
    assert len(logger) == 1
    assert (tmp_path / "failures.jsonl").exists()


def test_canonical_schema_type():
    """Canonical schema is a TypedDict and accepts the documented shape."""
    from src import canonical_schema
    from src.canonical_schema import CanonicalCell, CanonicalTable

    # TypedDicts carry their declared keys in __annotations__.
    assert "row_start" in CanonicalCell.__annotations__
    assert "is_header" in CanonicalCell.__annotations__
    assert "cells" in CanonicalTable.__annotations__

    # A dict matching the schema is valid at runtime (TypedDict is structural).
    cell: CanonicalCell = {
        "row_start": 0, "row_end": 1,
        "col_start": 0, "col_end": 1,
        "text": "Total", "is_header": True,
    }
    table: CanonicalTable = {"num_rows": 1, "num_cols": 1, "cells": [cell]}
    assert table["num_rows"] == 1
    assert table["cells"][0]["text"] == "Total"
    assert canonical_schema.TEXT_SOURCE_GT == "gt"
