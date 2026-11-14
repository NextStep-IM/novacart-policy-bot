"""Offline contract tests for the deterministic Step 3 calculator."""

from decimal import Decimal

import pytest

from src.calculator import (
    add_business_days,
    calculate_checkout,
    calculate_prorated_warranty_refund,
    calculate_return_refund,
    compute_international_export_price,
    count_full_months,
    extract_gst_from_inclusive,
    round_half_up,
    split_amount_proportionally,
)


def test_rounding_and_gst():
    assert round_half_up(Decimal("2.5")) == 3
    assert extract_gst_from_inclusive(11800) == 1800


def test_date_rules():
    assert count_full_months("2026-01-10", "2026-04-09") == 2
    assert count_full_months("2026-01-10", "2026-04-10") == 3
    assert count_full_months("2026-01-31", "2026-02-28") == 1
    assert add_business_days("2026-01-09", 1) == "2026-01-12"


def test_proportional_and_policy_examples():
    assert split_amount_proportionally(500, [5000, 5000, 5500]) == [161, 161, 178]
    assert split_amount_proportionally(3333, [2500, 1000, 500]) == [2083, 833, 417]
    assert compute_international_export_price(11800) == 10000


def test_checkout_examples():
    metro = calculate_checkout(
        items=[{"listed_price": 5200, "category": "electronics"}],
        promo_code="WELCOME10",
        shipping_fee=250,
    )
    assert metro["step_7_total_payable"] == 4930
    international = calculate_checkout(
        items=[{"listed_price": 11800, "category": "electronics"}],
        region="international",
        destination="UAE",
        promo_code="WELCOME10",
    )
    assert international["step_7_total_payable"] == 12500


def test_return_and_warranty_examples():
    checkout = calculate_checkout(
        items=[
            {"listed_price": 2400, "category": "clothing"},
            {"listed_price": 3600, "category": "clothing"},
        ],
        promo_code="STYLE15",
        shipping_fee=0,
    )
    refund = calculate_return_refund(
        [{"paid_price": checkout["step_1_item_prices"][0] - checkout["step_3_discount_shares"][0], "category": "clothing"}],
        return_fee=200,
    )
    assert refund["refund_amount"] == 1840
    assert calculate_prorated_warranty_refund(24000, 12, "2026-01-10", "2026-04-12")["refund_amount"] == 18000


def test_invalid_inputs():
    with pytest.raises(ValueError):
        split_amount_proportionally(10, [])
    with pytest.raises(ValueError):
        add_business_days("not-a-date", 1)
