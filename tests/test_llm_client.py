"""LLM client tests (CPU, no API key) - Phase 1C.

A fake completer exercises the full prompt-build -> parse path without Gemini (P3).
"""

from src.llm_client import LLMAnswer, build_prompt, generate_answer, parse_answer

_EVIDENCE = [
    {"chunk_id": "table:a", "text": "Revenue: 2018 = 13,223"},
    {"chunk_id": "table:b", "text": "Cash: 2018 = 500"},
]
_IDS = ["table:a", "table:b"]


def test_build_prompt_includes_question_ids_and_text():
    prompt = build_prompt("What was Revenue in 2018?", _EVIDENCE)
    assert "What was Revenue in 2018?" in prompt
    assert "table:a" in prompt and "table:b" in prompt
    assert "13,223" in prompt
    assert "JSON" in prompt  # the model is told to return JSON


def test_parse_clean_json():
    raw = '{"answer": "13,223", "citations": ["table:a"], "abstained": false}'
    ans = parse_answer(raw, _IDS)
    assert ans == LLMAnswer(answer="13,223", citations=["table:a"], abstained=False, raw=raw)


def test_parse_strips_markdown_fences():
    raw = '```json\n{"answer": "500", "citations": ["table:b"], "abstained": false}\n```'
    ans = parse_answer(raw, _IDS)
    assert ans.answer == "500"
    assert ans.citations == ["table:b"]


def test_parse_filters_unknown_citation():
    raw = '{"answer": "x", "citations": ["table:a", "table:ZZZ"], "abstained": false}'
    ans = parse_answer(raw, _IDS)
    assert ans.citations == ["table:a"]  # hallucinated id dropped


def test_parse_abstain():
    raw = '{"answer": "", "citations": [], "abstained": true}'
    ans = parse_answer(raw, _IDS)
    assert ans.abstained is True
    assert ans.answer == ""


def test_parse_non_json_falls_back_to_raw_text():
    ans = parse_answer("13,223", _IDS)
    assert ans.answer == "13,223"
    assert ans.citations == []
    assert ans.abstained is False


def test_generate_answer_with_fake_completer():
    def fake_complete(prompt):
        assert "Revenue" in prompt  # the prompt reached the completer
        return '{"answer": "13,223", "citations": ["table:a"], "abstained": false}'

    ans = generate_answer("What was Revenue in 2018?", _EVIDENCE, complete=fake_complete)
    assert ans.answer == "13,223"
    assert ans.citations == ["table:a"]
