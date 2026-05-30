"""Financial number normalization (DESIGN_SPEC §5.13).

V9 fix: looks_numeric() requires at least one digit (or a pure dash) before OCR
character substitutions (O->0, I->1) are applied, so text like "Operating Income
(Loss)" is not mistaken for a number because of its parentheses.
"""

import re
from typing import Optional


def looks_numeric(raw: str) -> bool:
    """
    Conservative: must contain at least one digit, or be a pure dash.
    Parentheses alone do NOT make a string numeric.
    """
    s = raw.strip()
    if s in ("-", "–", "—"):
        return True
    return bool(re.search(r'\d', s))


def normalize_financial_number(
    raw: str,
    dash_as_zero: bool = True,
    percent_as_ratio: bool = True
) -> Optional[float]:
    """
    Normalize financial number string to float.
    Returns None if not numeric.

    dash_as_zero: dash -> 0.0 (True) or None (False)
    percent_as_ratio: 12.5% -> 0.125 (True) or 12.5 (False)
    """
    if not raw or not raw.strip():
        return None

    s = raw.strip()
    s = re.sub(r'[$£€¥]', '', s).strip()

    if s in ('-', '–', '—', '- ', ' -'):
        return 0.0 if dash_as_zero else None

    is_negative = False
    if s.startswith('(') and s.endswith(')'):
        is_negative = True
        s = s[1:-1].strip()

    is_percent = False
    if s.endswith('%'):
        is_percent = True
        s = s[:-1].strip()

    # OCR substitutions — ONLY if string contains a digit
    if looks_numeric(s):
        s = s.replace('O', '0').replace('o', '0')
        s = s.replace('l', '1').replace('I', '1')

    s = s.replace(' ', '').replace(',', '')

    try:
        value = float(s)
    except ValueError:
        return None

    if is_negative:
        value = -value
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
