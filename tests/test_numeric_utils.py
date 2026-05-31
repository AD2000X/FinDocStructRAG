"""Numeric normalization tests (DESIGN_SPEC §13).

The key V9 regression: parentheses without digits must NOT count as numeric, so OCR
character substitutions are not applied to text like "Operating Income (Loss)".
"""

from src.numeric_utils import (
    looks_numeric,
    normalize_cell_text,
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


# --- Phase 1B financial-cell formatting -----------------------------------

def test_normalize_strips_dot_leaders():
    # FinTabNet cells carry a dotted leader before the value.
    assert normalize_financial_number(". . . . . . $ 45,854") == 45854.0


def test_normalize_currency_letter_confusion():
    # OCR misreads $ as S, or trails a stray letter; the single number still resolves.
    assert normalize_financial_number("409,110 S") == 409110.0
    assert normalize_financial_number("S 784,209") == 784209.0
    assert normalize_financial_number("158,389 A") == 158389.0


def test_normalize_rejects_multiple_numbers():
    # Two columns merged into one cell (a spatial error): not a single number, so unmatched.
    assert normalize_financial_number("2011 2010") is None
    assert normalize_financial_number("$10,376 $ 9,812") is None


def test_normalize_preserves_v9_operating_income_loss():
    assert normalize_financial_number("Operating Income (Loss)") is None


def test_relaxed_match_dot_leader_vs_plain():
    assert relaxed_numeric_match("45,854", ". . . . . . $ 45,854") is True


def test_normalize_cell_text_strips_leaders():
    assert normalize_cell_text("Purchased technology . . . .") == "Purchased technology"
    assert normalize_cell_text("Purchased technology.") == "Purchased technology"


def test_normalize_cell_text_keeps_decimal_and_currency():
    assert normalize_cell_text("5.00%") == "5.00%"
    assert normalize_cell_text(". . . . $ 45,854") == "$ 45,854"


# --- Word-level OCR token spacing (return_word_box=True) -------------------

def test_normalize_rejoins_separator_spaced_digit_groups():
    # Word-level tokens joined with spaces: separators end up space-padded.
    assert normalize_financial_number("$ 13 , 223") == 13223.0
    assert normalize_financial_number("5 , 483") == 5483.0
    assert normalize_financial_number("$ ( 250 , 721 )") == -250721.0
    assert normalize_financial_number("131 , 225") == 131225.0
    # multi-group thousands value
    assert normalize_financial_number("1 , 234 , 567") == 1234567.0


def test_normalize_rejoin_does_not_mask_merged_columns():
    # A space NOT flanking a separator is a real two-column merge; still rejected.
    assert normalize_financial_number("2011 2010") is None
    assert normalize_financial_number("$10,376 $ 9,812") is None


def test_relaxed_match_word_spaced_vs_gt():
    assert relaxed_numeric_match("$ 13 , 223", "$13,223") is True
    assert relaxed_numeric_match("$ ( 250 , 721 )", "$ (250,721 )") is True
