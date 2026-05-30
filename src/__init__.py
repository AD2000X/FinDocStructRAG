"""FinDocStructRAG source package.

Kept import-light on purpose: importing `src` (or its Phase 0 modules) must not pull
in GPU dependencies (torch / transformers / paddleocr), so the local CPU test loop and
`from src import ...` work without the Colab stack installed.
"""
