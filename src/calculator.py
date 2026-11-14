"""Deterministic NovaCart money and date calculations.

The module is deliberately independent of policy-file I/O, network clients and
the system clock.  Callers pass the policy values selected by the resolver.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

PROMO_RULES = {
    "WELCOME10": {"percent": 10, "minimum": 3000, "cap": 1000},
    "STYLE15": {"percent": 15, "minimum": 4000, "cap": 1500},
    "TECH500": {"flat": 500, "minimum": 15000, "cap": 500},
}
STORE_CREDIT_BONUS = Decimal("0.05")


def _decimal(value: Decimal | int | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def round_half_up(value: Decimal | int | str) -> int:
    """Round a monetary value to whole PKR (00_how_policies_work_together.md, Section 1.3)."""
    return int(_decimal(value).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def extract_gst_from_inclusive(inclusive_price_pkr: int) -> int:
    """Extract 18/118 of an inclusive price (00_how_policies_work_together.md, Section 1.2)."""
    if inclusive_price_pkr < 0:
        raise ValueError("inclusive_price_pkr must be non-negative")
    return round_half_up(_decimal(inclusive_price_pkr) * Decimal(18) / Decimal(118))


def compute_international_export_price(listed_price_pkr: int) -> int:
    """Remove GST per item before summing (20_international_addendum.md, Section 3.1)."""
    if listed_price_pkr < 0:
        raise ValueError("listed_price_pkr must be non-negative")
    return round_half_up(_decimal(listed_price_pkr) / Decimal("1.18"))


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO date: {value!r}") from exc


def add_calendar_days(start_date: str, days: int) -> str:
    """Add calendar days (00_how_policies_work_together.md, Section 1.6)."""
    return (_date(start_date) + timedelta(days=days)).isoformat()


def add_business_days(start_date: str, business_days: int) -> str:
    """Add Monday-Friday business days, excluding no holidays (00_how_policies_work_together.md, Section 1.5)."""
    if business_days < 0:
        raise ValueError("business_days must be non-negative")
    current = _date(start_date)
    for _ in range(business_days):
        current += timedelta(days=1)
        while current.weekday() > 4:
            current += timedelta(days=1)
    return current.isoformat()


def calendar_days_between(start_date: str, end_date: str) -> int:
    """Return elapsed calendar days, rejecting reversed dates (00_how_policies_work_together.md, Section 1.6)."""
    start, end = _date(start_date), _date(end_date)
    if end < start:
        raise ValueError("end_date must not precede start_date")
    return (end - start).days


def count_full_months(start_date: str, end_date: str) -> int:
    """Count complete calendar months using the anniversary rule (00_how_policies_work_together.md, Section 1.7)."""
    start, end = _date(start_date), _date(end_date)
    if end < start:
        return 0
    months = (end.year - start.year) * 12 + end.month - start.month
    anniversary = min(start.day, calendar.monthrange(end.year, end.month)[1])
    return max(0, months - (end.day < anniversary))


def split_amount_proportionally(total_to_split: int, item_weights: list[int]) -> list[int]:
    """Split and allocate rounding remainder to the highest-priced item (12_promo_code_rules.md, Sections 6.1-6.2)."""
    if not item_weights:
        raise ValueError("item_weights must not be empty")
    if total_to_split < 0 or any(weight < 0 for weight in item_weights):
        raise ValueError("amount and weights must be non-negative")
    total_weight = sum(item_weights)
    if total_weight == 0:
        raise ValueError("item_weights must have a positive sum")
    shares = [
        round_half_up(_decimal(total_to_split) * _decimal(weight) / _decimal(total_weight))
        for weight in item_weights
    ]
    remainder = total_to_split - sum(shares)
    if remainder:
        index = max(range(len(item_weights)), key=lambda i: (item_weights[i], -i))
        shares[index] += remainder
    return shares


def _item_price(item: dict[str, Any]) -> int:
    if item.get("sale_price") is not None:
        return int(item["sale_price"])
    if item.get("price") is not None:
        return int(item["price"])
    return int(item["listed_price"])


def _promo_eligible(code: str, item: dict[str, Any]) -> bool:
    category = str(item.get("category", "")).lower()
    if item.get("marketplace") or item.get("seller") == "marketplace":
        return False
    if item.get("final_sale") or item.get("bundle_id"):
        return False
    if code == "STYLE15":
        return category == "clothing"
    if code == "TECH500":
        return category == "electronics"
    return not category.startswith("grocery") and category != "perishables"


def _shipping_values(region: str, after_discount: int, order_date: str | None, version: str | None) -> tuple[int, int, int]:
    if region == "international":
        return 3500, 0, 0
    is_2027 = version in {"2027", "v2027"} or (
        order_date is not None and _date(order_date) >= date(2027, 2, 1)
    )
    if region == "remote":
        return (0 if after_discount >= 12000 else 600), 0, 0
    if region == "rest_of_pakistan":
        return (0 if after_discount >= (8500 if is_2027 else 7500) else (400 if is_2027 else 350)), 0, (3000 if is_2027 else 2500)
    return (0 if after_discount >= (6000 if is_2027 else 5000) else (300 if is_2027 else 250)), (600 if is_2027 else 500), (1800 if is_2027 else 1500)


def calculate_checkout(
    *,
    items: list[dict[str, Any]],
    region: str = "metro",
    promo_code: str | None = None,
    shipping_speed: str = "standard",
    payment_method: str | None = None,
    order_date: str | None = None,
    policy_version: str | None = None,
    destination: str | None = None,
    first_order: bool | None = None,
    cod_order_total: int | None = None,
    shipping_fee: int | None = None,
    shipping_schedule: list[dict[str, int]] | None = None,
    freeship: bool = False,
    express_fee_override: int | None = None,
    bulky_surcharge_per_item: int | None = None,
    extra_charges: dict[str, int] | None = None,
    bundle_prices: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Calculate the seven checkout steps (00_how_policies_work_together.md, Section 5.1)."""
    if not items:
        raise ValueError("items must not be empty")
    if order_date is not None:
        _date(order_date)
    prices = [_item_price(item) * int(item.get("quantity", 1)) for item in items]
    bundle_groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        if item.get("bundle_id"):
            bundle_groups.setdefault(str(item["bundle_id"]), []).append(index)
    for bundle_id, indexes in bundle_groups.items():
        bundle_price = (bundle_prices or {}).get(bundle_id, items[indexes[0]].get("bundle_price"))
        if bundle_price is not None:
            shares = split_amount_proportionally(int(bundle_price), [int(items[i]["listed_price"]) for i in indexes])
            for index, share in zip(indexes, shares):
                prices[index] = share
    if region == "international":
        prices = [compute_international_export_price(price) for price in prices]
    subtotal = sum(prices)
    code = (promo_code or "").upper() or None
    discount = 0
    discount_shares = [0] * len(items)
    promo_reason = "No discount code supplied."
    if code and code != "FREESHIP":
        rule = PROMO_RULES.get(code)
        if rule is None:
            promo_reason = f"{code} rejected: unknown code."
        elif code == "WELCOME10" and first_order is False:
            promo_reason = "WELCOME10 rejected: it is limited to a first order."
        else:
            eligible = [i for i, item in enumerate(items) if _promo_eligible(code, item)]
            eligible_total = sum(prices[i] for i in eligible)
            if eligible_total < rule["minimum"]:
                promo_reason = f"{code} rejected: eligible subtotal Rs. {eligible_total} is below Rs. {rule['minimum']}."
            elif code == "WELCOME10" and order_date and order_date > "2026-12-31":
                promo_reason = "WELCOME10 rejected: its last order date has passed."
            elif code == "STYLE15" and order_date and order_date > "2026-08-31":
                promo_reason = "STYLE15 rejected: its last order date has passed."
            elif code == "TECH500" and order_date and order_date > "2027-01-31":
                promo_reason = "TECH500 rejected: its last order date has passed."
            else:
                discount = min(
                    rule.get("cap", rule.get("flat", 0)),
                    round_half_up(_decimal(eligible_total) * _decimal(rule["percent"]) / 100)
                    if "percent" in rule else eligible_total,
                )
                shares = split_amount_proportionally(discount, [prices[i] for i in eligible])
                for index, share in zip(eligible, shares):
                    discount_shares[index] = share
                promo_reason = f"{code} applied to eligible subtotal Rs. {eligible_total}."
    after_discount = subtotal - discount
    standard_shipping, express_fee, bulky_fee = _shipping_values(region, after_discount, order_date, policy_version)
    if shipping_fee is not None:
        standard_shipping = int(shipping_fee)
    if express_fee_override is not None:
        express_fee = int(express_fee_override)
    if bulky_surcharge_per_item is not None:
        bulky_fee = int(bulky_surcharge_per_item)
    if shipping_schedule and region != "international":
        eligible = [tier for tier in shipping_schedule if int(tier["min_subtotal"]) <= after_discount]
        if not eligible:
            raise ValueError("shipping_schedule has no applicable tier")
        standard_shipping = int(max(eligible, key=lambda tier: int(tier["min_subtotal"]))["fee"])
    if region == "international":
        if not destination and shipping_fee is None:
            return {"missing_inputs": ["destination"], "step_1_item_prices": prices,
                    "step_2_merchandise_subtotal": subtotal, "step_3_discount": discount}
        international_fees = {"uae": 3500, "united arab emirates": 3500, "uk": 5500, "united kingdom": 5500, "usa": 6500, "united states": 6500}
        if (destination or "").lower() not in international_fees and shipping_fee is None:
            return {"missing_inputs": ["destination (UAE, UK, or USA)"], "step_1_item_prices": prices,
                    "step_2_merchandise_subtotal": subtotal, "step_3_discount": discount}
        shipping = international_fees.get((destination or "").lower(), int(shipping_fee or 0))
        express_fee = 0
    elif shipping_speed.lower() == "express":
        if region != "metro":
            raise ValueError("express delivery is available only in Metro cities")
        shipping = express_fee
        express_fee = 0
    else:
        shipping = standard_shipping
        express_fee = 0
    freeship = (code == "FREESHIP" or freeship) and region in {"metro", "rest_of_pakistan"} and after_discount >= 2000 and shipping_speed.lower() == "standard"
    if freeship:
        shipping = 0
    bulky_total = bulky_fee * sum(int(item.get("quantity", 1)) for item in items if item.get("bulky") or item.get("category") == "furniture/bulky")
    cod = 100 if (payment_method or "").lower() in {"cod", "cash on delivery"} else 0
    extras = {"express_fee": express_fee, "bulky_surcharge": bulky_total, "cod_fee": cod}
    extras.update({key: int(value) for key, value in (extra_charges or {}).items()})
    total = after_discount + shipping + sum(extras.values())
    return {
        "steps": [
            {"step": 1, "amount": prices, "explanation": "Item prices after sale, bundle, or export pricing.", "citations": ["00_how_policies_work_together.md, Section 5.1", "13_sales_and_bundles.md, Section 1.2"]},
            {"step": 2, "amount": subtotal, "explanation": "Merchandise subtotal.", "citations": ["00_how_policies_work_together.md, Section 5.1"]},
            {"step": 3, "amount": discount, "explanation": promo_reason, "citations": ["12_promo_code_rules.md, Section 3.1"]},
            {"step": 4, "amount": after_discount, "explanation": "Subtotal after discount.", "citations": ["00_how_policies_work_together.md, Section 5.1"]},
            {"step": 5, "amount": shipping, "explanation": "Standard, express, remote, or international shipping.", "citations": ["06_shipping_fees_and_free_shipping.md, Section 2.2"]},
            {"step": 6, "amount": sum(extras.values()), "explanation": "Express, bulky, and COD charges.", "citations": ["00_how_policies_work_together.md, Section 5.1"]},
            {"step": 7, "amount": total, "explanation": "Total payable and GST component.", "citations": ["00_how_policies_work_together.md, Section 1.2"]},
        ],
        "step_1_item_prices": prices, "step_2_merchandise_subtotal": subtotal, "step_3_discount": discount,
        "step_3_discount_shares": discount_shares, "step_3_note": promo_reason,
        "step_4_subtotal_after_discount": after_discount, "step_5_shipping_fee": shipping,
        "step_5_freeship_applied": freeship, "step_6_extra_charges": extras,
        "step_6_extra_total": sum(extras.values()), "step_7_total_payable": total,
        "step_7_gst": 0 if region == "international" else extract_gst_from_inclusive(total),
        "conflict": ({"10_payments.md, Section 3.2": "Rs. 40,000", "11_cod_and_failed_payments.md, Section 1.2": "Rs. 50,000", "message": "NovaCart support must confirm which COD limit applies."}
                    if cod and (cod_order_total or total) > 40000 else None),
    }


def calculate_checkout_total(items: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper for the Step 2 API."""
    return calculate_checkout(items=items, **kwargs)


def calculate_return_refund(
    items: list[dict[str, Any]],
    *,
    delivery_date: str | None = None,
    return_date: str | None = None,
    reason: str = "change_of_mind",
    region: str = "metro",
    policy_version: str = "v3",
    return_fee: int | None = None,
    shipping_fee: int = 0,
    to_store_credit: bool = False,
    restocking_fee_percent: Decimal | int | str = 0,
    promo_discount: int = 0,
    bundle_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate an approved return refund (01_returns_policy.md, Sections 2-5)."""
    if delivery_date and return_date:
        if calendar_days_between(delivery_date, return_date) < 0:
            raise ValueError("return_date must not precede delivery_date")
    if not items:
        raise ValueError("items must not be empty")
    error_return = reason in {"defective_or_wrong", "novacart_error"}
    if delivery_date and return_date:
        windows = {
            "electronics": 10 if policy_version in {"v2", "2025"} else 7,
            "clothing": 14 if policy_version in {"v2", "2025"} else 30,
            "furniture/bulky": 14,
            "personal care": 14,
        }
        if region == "international":
            window = 10
        elif region == "remote":
            window = 14
        else:
            window = windows.get(str(items[0].get("category", "")).lower())
        if window is None:
            return {"eligible": False, "reason": "category has no specified return window", "refund_amount": 0}
        if not error_return and calendar_days_between(delivery_date, return_date) > window:
            return {"eligible": False, "reason": "return window has passed", "refund_amount": 0}
    if return_fee is None:
        if error_return or region == "international":
            return_fee = 0
        elif region == "remote":
            return_fee = 400
        elif any(item.get("category") == "furniture/bulky" or item.get("bulky") for item in items):
            return_fee = 2000 if policy_version not in {"v2", "2025"} else 1500
        else:
            return_fee = 200 if policy_version not in {"v2", "2025"} else 150
    lines: list[dict[str, Any]] = []
    bundle_shares: dict[int, int] = {}
    if bundle_items:
        bundle_total = int(bundle_items[0].get("bundle_price", 0))
        bundle_shares = dict(enumerate(split_amount_proportionally(
            bundle_total, [int(entry["listed_price"]) for entry in bundle_items]
        )))
    for item_index, item in enumerate(items):
        paid = int(item.get("paid_price", item.get("price", item.get("listed_price", 0))))
        if item.get("bundle_index") in bundle_shares:
            paid = bundle_shares[int(item["bundle_index"])]
        elif item.get("bundle_price") is not None:
            paid = int(item["bundle_price"])
        if promo_discount and len(items) > 1:
            promo_shares = split_amount_proportionally(
                promo_discount, [int(entry.get("paid_price", entry.get("price", entry.get("listed_price", 0)))) for entry in items]
            )
            paid -= promo_shares[item_index]
        percent = Decimal("0")
        if not error_return and item.get("restocking_fee_percent") is not None:
            percent = _decimal(item["restocking_fee_percent"])
        elif not error_return and restocking_fee_percent:
            percent = _decimal(restocking_fee_percent)
        elif not error_return and item.get("category") == "electronics" and item.get("opened"):
            percent = Decimal("15") if policy_version not in {"v2", "2025"} else Decimal("10")
        elif not error_return and item.get("category") == "furniture/bulky":
            percent = Decimal("10")
        fee = round_half_up(_decimal(paid) * percent / 100)
        lines.append({"name": item.get("name"), "paid_price": paid, "restocking_fee": fee, "net": paid - fee})
    refund = max(0, sum(line["net"] for line in lines) - int(return_fee))
    if not error_return and shipping_fee and reason not in {"cancel_before_dispatch", "lost_parcel"}:
        shipping_refunded = 0
    else:
        shipping_refunded = shipping_fee
    bonus = min(1000, round_half_up(_decimal(refund) * STORE_CREDIT_BONUS)) if to_store_credit else 0
    return {"steps": lines, "lines": lines, "return_fee": int(return_fee), "shipping_refund": shipping_refunded,
            "refund_amount": refund, "store_credit_bonus": bonus, "total_store_credit": refund + bonus if to_store_credit else 0,
            "policy_version": policy_version, "region": region}


def calculate_prorated_warranty_refund(paid_price: int, warranty_months: int, delivery_date: str, claim_date: str) -> dict[str, Any]:
    """Calculate the remaining warranty refund (08_warranty_policy.md, Section 4.3)."""
    if paid_price < 0 or warranty_months <= 0:
        raise ValueError("paid_price must be non-negative and warranty_months must be positive")
    months = count_full_months(delivery_date, claim_date)
    remaining = max(warranty_months - months, 0)
    return {"paid_price": paid_price, "warranty_months": warranty_months, "full_months_used": months,
            "months_remaining": remaining, "refund_amount": round_half_up(_decimal(paid_price) * remaining / warranty_months)}
