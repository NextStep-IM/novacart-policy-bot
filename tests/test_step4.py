"""Offline Step 4 hardening checks."""

from pathlib import Path
import re

from src.calculator import calculate_checkout, calculate_prorated_warranty_refund


def test_calculator_has_no_clock_or_float_arithmetic():
    source = Path("src/calculator.py").read_text(encoding="utf-8")
    assert not re.search(r"datetime\.now|date\.today|time\.time|float\(", source)


def test_checkout_requires_international_destination():
    result = calculate_checkout(
        items=[{"listed_price": 11800, "category": "electronics"}],
        region="international",
        promo_code="WELCOME10",
    )
    assert result["missing_inputs"] == ["destination"]


def test_warranty_worked_example_is_deterministic():
    result = calculate_prorated_warranty_refund(
        24000, 12, "2026-01-10", "2026-04-12"
    )
    assert result["refund_amount"] == 18000
