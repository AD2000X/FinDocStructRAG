"""LLM answer generation for table-only RAG (Phase 1C, P5).

The single place an LLM is used in the project: grounded answer generation over retrieved
table chunks. Retrieval never calls an LLM (P5). The eval consumes only the provider-neutral
`LLMAnswer`, never the SDK response object, so the provider can be swapped without touching
eval code.

The prompt build and the response parse are pure and unit-tested; the Gemini call is a lazy,
injectable `complete(prompt) -> str` so tests exercise the full path with a fake completer
and no API key (P3). The model is instructed to ground its answer in the provided tables,
cite the table ids it used, and abstain (set "abstained") when the answer is not present.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from . import config

# A rate-limit error carries a suggested wait ("Please retry in 36.4s").
_RETRY_RE = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)

# The model must reply with this JSON shape and nothing else.
_PROMPT_TEMPLATE = """You are a financial-report QA assistant. Answer the question using ONLY \
the tables below. Each table is marked with an id.

Rules:
- If the answer is in the tables, set "answer" to the exact value only (a number or short \
phrase), nothing else.
- List the id of every table you used in "citations".
- If the tables do not contain the answer, set "abstained" to true and "answer" to "".
- Reply with ONLY this JSON object, no prose and no markdown fences:
  {{"answer": "...", "citations": ["<table id>", ...], "abstained": false}}

Tables:
{evidence}

Question: {question}"""


@dataclass(frozen=True)
class LLMAnswer:
    """Provider-neutral answer the eval consumes (never the SDK response).

    answer: the extracted value (empty when abstained). citations: table chunk_ids the model
    used, filtered to ids actually in the evidence. abstained: the model reported the answer
    is not in the tables. raw: the model's raw text, kept only for debugging/traceability.
    """

    answer: str
    citations: list[str]
    abstained: bool
    raw: str = field(default="", repr=False)


def build_prompt(question: str, evidence: list[dict]) -> str:
    """Build the grounded-QA prompt from the question and retrieved chunks.

    evidence: chunks (dicts with "chunk_id" and "text") in rank order. Each is rendered with
    its id so the model can cite it.
    """
    blocks = [f"[id: {e['chunk_id']}]\n{e.get('text', '')}" for e in evidence]
    return _PROMPT_TEMPLATE.format(evidence="\n\n".join(blocks), question=question)


def _extract_json(raw: str) -> dict | None:
    """Best-effort: pull the first {...} object out of the model text (fences tolerated)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_answer(raw: str, evidence_ids: list[str]) -> LLMAnswer:
    """Parse the model text into an LLMAnswer.

    Citations are filtered to ids present in the evidence, so a hallucinated or mistyped id is
    dropped rather than counted. If the JSON cannot be parsed, fall back to the stripped raw
    text as the answer with no citations (not treated as an abstain), so a usable answer is
    not silently discarded.
    """
    known = set(evidence_ids)
    obj = _extract_json(raw)
    if obj is None:
        return LLMAnswer(answer=raw.strip(), citations=[], abstained=False, raw=raw)
    citations = [c for c in obj.get("citations", []) if c in known]
    return LLMAnswer(
        answer=str(obj.get("answer", "") or "").strip(),
        citations=citations,
        abstained=bool(obj.get("abstained", False)),
        raw=raw,
    )


def generate_answer(question: str, evidence: list[dict], complete=None) -> LLMAnswer:
    """Generate a grounded answer. complete(prompt) -> raw text; defaults to the Gemini call.

    Injecting complete keeps the model out of unit tests: a fake completer exercises the whole
    prompt-build -> parse path with no API key.
    """
    if complete is None:
        complete = build_gemini_complete()
    prompt = build_prompt(question, evidence)
    raw = complete(prompt)
    return parse_answer(raw, [e["chunk_id"] for e in evidence])


def _retry_delay_seconds(error_text: str, attempt: int) -> float:
    """Seconds to wait before retrying a rate-limited call.

    Prefer the server's suggested delay ("Please retry in 36.4s") plus a 1s margin; otherwise
    exponential backoff capped at 60s.
    """
    m = _RETRY_RE.search(error_text or "")
    if m:
        return float(m.group(1)) + 1.0
    return min(60.0, 5.0 * (2 ** attempt))


def build_gemini_complete(model_name: str | None = None, max_retries: int = 6):
    """Build the Gemini completer (Colab/API) -> callable prompt -> raw text. Lazy import.

    Reads the key from GEMINI_API_KEY or GOOGLE_API_KEY, and the model from GEMINI_MODEL or
    config.LLM_MODEL. google-generativeai is the single provider SDK (P5).

    Retries on a rate-limit error (429 ResourceExhausted), waiting the server's suggested
    delay, so a free-tier per-minute request cap throttles the run instead of crashing it.
    """
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (or GOOGLE_API_KEY) for answer generation")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name or os.environ.get("GEMINI_MODEL")
                                  or config.LLM_MODEL)

    def complete(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                return model.generate_content(prompt).text or ""
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(_retry_delay_seconds(str(e), attempt))
        return ""  # unreachable: the loop returns or raises

    return complete
