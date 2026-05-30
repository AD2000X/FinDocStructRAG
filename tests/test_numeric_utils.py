"""Numeric normalization tests (DESIGN_SPEC §13).

The key V9 regression: parentheses without digits must NOT count as numeric, so OCR
character substitutions are not applied to text like "Operating Income (Loss)".
"""

from src.numeric_utils import (
    looks_numeric,
    normalize_financial_number,
    relaxed_numeric_match,
)


def test_looks_numeric_with_digits():
    assert looks_numeric("1,234") is True


def test_looks_numeric_pure_dash():
    assert looks_numeric("—") is True


def test_looks_numeric_parentheses_no_digits():
    assert looks_numeric("Operating Income (Loss)") is False


def test_looks_numeric_parentheses_with_digits():
    assert looks_numeric("(1,234)") is True


def test_looks_numeric_plain_text():
    assert looks_numeric("Total Assets") is False


def test_ocr_sub_not_applied_to_text():
    # "Operating" must not have its 'O'/'I'/'l' rewritten to digits, and the string
    # is not a number, so normalization returns None.
    assert normalize_financial_number("Operating") is None


def test_normalize_parentheses_negative():
    assert normalize_financial_number("(1,234)") == -1234.0


def test_normalize_dash_as_zero():
    assert normalize_financial_number("—") == 0.0
    assert normalize_financial_number("—", dash_as_zero=False) is None


def test_normalize_percent_as_ratio():
    assert normalize_financial_number("12.5%") == 0.125
    assert normalize_financial_number("12.5%", percent_as_ratio=False) == 12.5


def test_relaxed_numeric_match_within_tolerance():
    assert relaxed_numeric_match("1,234", "$1,234.00") is True
    assert relaxed_numeric_match("1,234", "2,000") is False
