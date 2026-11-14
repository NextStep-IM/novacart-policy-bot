"""Step 4: pure deterministic money and date arithmetic.

Everything uses decimal.Decimal with ROUND_HALF_UP to the whole rupee, applied at
EVERY intermediate step (00_how_policies_work_together.md, Section 1.1-1.7, 5.1).
No function here reads the wall clock.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

_ONE = Decimal("1")

PROMO_RULES: dict[str, dict[str, Any]] = {
    "WELCOME10": {"percent": 10, "min_eligible": 3000, "max_discount": 1000},
    "STYLE15": {"percent": 15, "min_eligible": 4000, "max_discount": 1500},
    "TECH500": {"flat": 500, "min_eligible": 15000},
}
FREESHIP_MIN_AFTER_DISCOUNT = 2000
FREESHIP_REGIONS = ("metro", "rest_of_pakistan")
COD_FEE = 100
STORE_CREDIT_BONUS_RATE = Decimal("0.05")
STORE_CREDIT_BONUS_CAP = 1000


def _d(value: Decimal | int | float | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def round_half_up(value: Decimal | int | float | str) -> int:
    """Round to the nearest whole rupee, halves away from zero (Section 1.1)."""
    return int(_d(value).quantize(_ONE, rounding=ROUND_HALF_UP))


def extract_gst_from_inclusive(price: int) -> int:
    """GST contained in a GST-inclusive price: round(price * 18 / 118)."""
    return round_half_up(Decimal(price) * 18 / 118)


def compute_international_export_price(listed_price: int) -> int:
    """GST-free export price: round(listed / 1.18) per item."""
    return round_half_up(Decimal(listed_price) / Decimal("1.18"))


def _to_date(value: str) -> date:
    return date.fromisoformat(value)


def add_calendar_days(start_date: str, days: int) -> str:
    """start_date + N calendar days (Section 1.6, inclusive of the deadline day)."""
    return (_to_date(start_date) + timedelta(days=days)).isoformat()


def add_business_days(start_date: str, business_days: int) -> str:
    """Mon-Fri only, no holidays; counting starts on the first business day AFTER the event."""
    if business_days < 0:
        raise ValueError("business_days must be >= 0")
    current = _to_date(start_date)
    counted = 0
    while counted < business_days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            counted += 1
    return current.isoformat()


def count_full_months(date_a: str, date_b: str) -> int:
    """Whole calendar months from date A to date B (Section 1.7)."""
    a, b = _to_date(date_a), _to_date(date_b)
    if b < a:
        return 0
    months = (b.year - a.year) * 12 + (b.month - a.month)
    anniversary_day = min(a.day, calendar.monthrange(b.year, b.month)[1])
    if b.day < anniversary_day:
        months -= 1
    return max(months, 0)


def split_amount_proportionally(total_to_split: int, weights: list[int]) -> list[int]:
    """share_i = round(total * w_i / sum(w)); the rounding difference goes to the
    highest-weight entry (first listed on a tie)."""
    if not weights:
        return []
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive number")
    shares = [round_half_up(Decimal(total_to_split) * w / weight_sum) for w in weights]
    diff = total_to_split - sum(shares)
    if diff:
        top = max(range(len(weights)), key=lambda i: (weights[i], -i))
        shares[top] += diff
    return shares


# ---------------------------------------------------------------------------
# Checkout (Section 5.1)
# ---------------------------------------------------------------------------
def _promo_eligible(code: str, item: dict[str, Any]) -> bool:
    category = str(item.get("category", "")).lower()
    if code == "WELCOME10":
        return not (
            category.startswith("grocery") or category == "perishables"
            or item.get("bundle_id") or item.get("final_sale") or item.get("marketplace")
        )
    if code == "STYLE15":
        return category == "clothing"
    if code == "TECH500":
        return category == "electronics"
    return False


def _pick_shipping(after_discount: int, schedule: list[dict[str, int]]) -> int:
    usable = [s for s in schedule if s["min_subtotal"] <= after_discount]
    if not usable:
        raise ValueError("shipping_schedule has no tier for this subtotal")
    return max(usable, key=lambda s: s["min_subtotal"])["fee"]


def calculate_checkout_total(
    items: list[dict[str, Any]],
    region: str = "metro",
    promo_code: str | None = None,
    payment_method: str | None = None,
    shipping_fee: int | None = None,
    shipping_schedule: list[dict[str, int]] | None = None,
    freeship: bool = False,
    express_fee: int = 0,
    bulky_surcharge_per_item: int = 0,
    extra_charges: dict[str, int] | None = None,
    bundle_prices: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Seven-step checkout breakdown.

    items: dicts with name, listed_price, optional sale_price, category,
    bundle_id, final_sale, marketplace, bulky.
    Fee amounts (shipping tiers, express, bulky, remote surcharges) come from the
    ACTIVE policy text (2026 file 06, 2027 file 21, file 19, file 20); pass either
    shipping_fee or shipping_schedule=[{"min_subtotal": int, "fee": int}, ...].
    For region == "international" each item price is converted to the GST-free
    export price (file 20, Section 3.1) before step 2.
    """
    bundle_prices = bundle_prices or {}

    # Step 1: item prices (sale price, or bundle share by list price)
    prices: list[int] = []
    for item in items:
        prices.append(int(item.get("sale_price") if item.get("sale_price") is not None else item["listed_price"]))
    for bundle_id, bundle_price in bundle_prices.items():
        idx = [i for i, it in enumerate(items) if it.get("bundle_id") == bundle_id]
        if not idx:
            continue
        shares = split_amount_proportionally(bundle_price, [int(items[i]["listed_price"]) for i in idx])
        for i, share in zip(idx, shares):
            prices[i] = share
    if region == "international":
        prices = [compute_international_export_price(p) for p in prices]

    # Step 2: merchandise subtotal
    subtotal = sum(prices)

    # Step 3: discount code on the eligible subtotal
    code = (promo_code or "").upper() or None
    discount = 0
    discount_shares = [0] * len(items)
    promo_note = "no promo code"
    if code and code != "FREESHIP":
        rule = PROMO_RULES.get(code)
        if rule is None:
            raise ValueError(f"unknown promo code {code}")
        elig_idx = [i for i, it in enumerate(items) if _promo_eligible(code, it)]
        eligible_subtotal = sum(prices[i] for i in elig_idx)
        if eligible_subtotal < rule["min_eligible"]:
            promo_note = f"{code}: eligible subtotal {eligible_subtotal} below minimum {rule['min_eligible']}"
        else:
            if "percent" in rule:
                raw = round_half_up(Decimal(eligible_subtotal) * rule["percent"] / 100)
                discount = min(raw, rule["max_discount"])
            else:
                discount = min(rule["flat"], eligible_subtotal)
            shares = split_amount_proportionally(discount, [prices[i] for i in elig_idx])
            for i, share in zip(elig_idx, shares):
                discount_shares[i] = share
            promo_note = f"{code} applied to eligible subtotal {eligible_subtotal}"
    free_ship_code = code == "FREESHIP" or freeship

    # Step 4: subtotal after discount
    after_discount = subtotal - discount

    # Step 5: standard shipping fee
    freeship_applied = False
    if free_ship_code and region in FREESHIP_REGIONS and after_discount >= FREESHIP_MIN_AFTER_DISCOUNT:
        shipping, freeship_applied = 0, True
    elif shipping_fee is not None:
        shipping = int(shipping_fee)
    elif shipping_schedule:
        shipping = _pick_shipping(after_discount, shipping_schedule)
    else:
        raise ValueError("provide shipping_fee or shipping_schedule from the active shipping policy")

    # Step 6: extra charges
    bulky_count = sum(1 for it in items if it.get("bulky"))
    cod = COD_FEE if (payment_method or "").lower() in ("cod", "cash on delivery") else 0
    extras = {
        "express_fee": int(express_fee),
        "bulky_surcharge": int(bulky_surcharge_per_item) * bulky_count,
        "cod_fee": cod,
        **{k: int(v) for k, v in (extra_charges or {}).items()},
    }
    extras_total = sum(extras.values())

    # Step 7: total payable
    total = after_discount + shipping + extras_total
    return {
        "region": region,
        "step_1_item_prices": prices,
        "step_2_merchandise_subtotal": subtotal,
        "step_3_discount": discount,
        "step_3_discount_shares": discount_shares,
        "step_3_note": promo_note,
        "step_4_subtotal_after_discount": after_discount,
        "step_5_shipping_fee": shipping,
        "step_5_freeship_applied": freeship_applied,
        "step_6_extra_charges": extras,
        "step_6_extra_total": extras_total,
        "step_7_total_payable": total,
    }


# ---------------------------------------------------------------------------
# Returns and warranty
# ---------------------------------------------------------------------------
def calculate_return_refund(
    items: list[dict[str, Any]],
    return_fee: int = 0,
    restocking_fee_percent: Decimal | int | float | str = 0,
    to_store_credit: bool = False,
    policy_version: str = "v3",
    region: str = "metro",
) -> dict[str, Any]:
    """refund = max(0, sum(paid_price - restocking_fee) - one return/pickup fee).

    The same formula applies to the 2026 edition (file 01, v3) and the 2025 edition
    (file 02, v2); the RATES differ, so pass the rates from the edition that the
    resolver marked active (and from file 19 / file 20 for those regions).
    items: dicts with paid_price (after any promo share) and optionally
    restocking_fee (flat rupees) or restocking_fee_percent.
    """
    lines = []
    for item in items:
        paid = int(item["paid_price"])
        if item.get("restocking_fee") is not None:
            fee = int(item["restocking_fee"])
        else:
            pct = _d(item.get("restocking_fee_percent", restocking_fee_percent))
            fee = round_half_up(Decimal(paid) * pct / 100)
        lines.append({"name": item.get("name"), "paid_price": paid,
                      "restocking_fee": fee, "net": paid - fee})
    net_total = sum(line["net"] for line in lines)
    refund = max(0, net_total - int(return_fee))
    bonus = min(STORE_CREDIT_BONUS_CAP, round_half_up(Decimal(refund) * STORE_CREDIT_BONUS_RATE)) if to_store_credit else 0
    return {
        "policy_version": policy_version,
        "region": region,
        "lines": lines,
        "net_after_restocking": net_total,
        "return_fee": int(return_fee),
        "refund_amount": refund,
        "store_credit_bonus": bonus,
        "total_store_credit": refund + bonus if to_store_credit else 0,
    }


def calculate_prorated_warranty_refund(
    paid_price: int, warranty_months: int, delivery_date: str, claim_date: str
) -> dict[str, Any]:
    """round(paid * (W - m) / W) where m = full months from delivery to claim."""
    if warranty_months <= 0:
        raise ValueError("warranty_months must be positive")
    months_used = count_full_months(delivery_date, claim_date)
    remaining = max(warranty_months - months_used, 0)
    refund = round_half_up(Decimal(paid_price) * remaining / warranty_months)
    return {
        "paid_price": paid_price,
        "warranty_months": warranty_months,
        "full_months_used": months_used,
        "months_remaining": remaining,
        "refund_amount": refund,
    }
