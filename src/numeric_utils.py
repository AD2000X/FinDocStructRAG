"""Financial number normalization (DESIGN_SPEC §5.13).

V9 fix: looks_numeric() requires at least one digit (or a pure dash) before OCR
character substitutions (O->0, I->1) are applied, so text like "Operating Income
(Loss)" is not mistaken for a number because of its parentheses.

Phase 1B financial-cell handling: normalize_financial_number tolerates the formatting
real FinTabNet cells carry - dot leaders ("Label . . . . 45,854"), a currency $ misread
as a letter, a stray trailing letter - by extracting the single numeric token. A cell
holding two or more numbers (e.g. two adjacent columns merged by a spatial error) is left
unmatched on purpose, so those errors are not papered over.
"""

import re
from typing import Optional

# Runs of 2+ dots (optionally space-separated): table leader dots, not a decimal point.
_LEADER_DOTS = re.compile(r"(?:\.\s*){2,}")
# A single number token: digits with optional thousands commas and a decimal part.
_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def looks_numeric(raw: str) -> bool:
    """
    Conservative: must contain at least one digit, or be a pure dash.
    Parentheses alone do NOT make a string numeric.
    """
    s = raw.strip()
    if s in ("-", "–", "—"):
        return True
    return bool(re.search(r'\d', s))


def normalize_cell_text(text: str) -> str:
    """Normalize cell text for comparison: collapse whitespace and drop leader dots.

    Leader dots (the dotted line linking a label to its value) are formatting, not
    content, so a run of 2+ dots is removed and residual leading/trailing dots stripped.
    A lone decimal point inside a number is one dot and is preserved.
    """
    s = " ".join((text or "").split())
    s = _LEADER_DOTS.sub(" ", s)
    s = s.strip(" .")
    return " ".join(s.split())


def normalize_financial_number(
    raw: str,
    dash_as_zero: bool = True,
    percent_as_ratio: bool = True
) -> Optional[float]:
    """
    Normalize a financial number string to float. Returns None if not a single number.

    dash_as_zero: dash -> 0.0 (True) or None (False)
    percent_as_ratio: 12.5% -> 0.125 (True) or 12.5 (False)
    """
    if not raw or not raw.strip():
        return None

    s = raw.strip()
    if s in ('-', '–', '—', '- ', ' -'):
        return 0.0 if dash_as_zero else None

    # Drop dot-leader runs (decimals, a single dot between digits, are not matched).
    s = _LEADER_DOTS.sub(' ', s)

    # Require exactly one numeric token. Zero -> not a number; two or more -> likely two
    # columns merged into one cell (a spatial extraction error), which we leave unmatched
    # rather than silently pick one of.
    if len(_NUMBER_TOKEN.findall(s)) != 1:
        return None

    is_negative = '(' in s and ')' in s
    is_percent = '%' in s

    # OCR substitutions are safe here: the cell is one number with at most stray markers
    # (a misread $ -> S, a trailing letter), which the digit-only filter below removes.
    s = s.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
    s = re.sub(r'[^0-9.\-]', '', s.replace(',', ''))

    try:
        value = float(s)
    except ValueError:
        return None

    if is_negative:
        value = -abs(value)
    if is_percent and percent_as_ratio:
        value = value / 100.0

    return value


def relaxed_numeric_match(
    pred: str, gt: str,
    tolerance: float = 0.01,
    dash_as_zero: bool = True,
    percent_as_ratio: bool = True
) -> bool:
    pred_val = normalize_financial_number(pred, dash_as_zero, percent_as_ratio)
    gt_val = normalize_financial_number(gt, dash_as_zero, percent_as_ratio)
    if pred_val is None or gt_val is None:
        return False
    if gt_val == 0:
        return abs(pred_val) < 1e-6
    return abs(pred_val - gt_val) / abs(gt_val) < tolerance
