"""Step 2: decide which policy files are in force for this customer/date/region.

Implements 00_how_policies_work_together.md Sections 3.1-3.2 and 4.1-4.5.
Never reads the wall clock: every date comes from the customer's message or
from policy front matter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import (
    ActivePolicyFile,
    ExcludedPolicyFile,
    ExtractedQueryContext,
    MandatoryDisclosure,
    PolicyResolutionResult,
)

# Files that cover the same topic with the same scope but different editions.
SUPERSESSION_GROUPS: dict[str, tuple[str, ...]] = {
    "returns_policy": ("01", "02"),
    "shipping_fees": ("06", "21"),
}
REGIONAL_FILES: dict[str, str] = {"19": "remote", "20": "international"}
PROMO_FILE = "12"
COD_CONFLICT_FILES = ("10", "11")
COD_CONFLICT_START = "2026-05-01"
SCOPE_RANK = {"general": 0, "specific": 1, "regional": 2}


def load_policy_documents(path: Path = Path("processed/policy_documents.json")) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):  # tolerate {"documents": [...]}
        data = data.get("documents", [])
    return list(data)


def _meta(doc: dict[str, Any]) -> dict[str, Any]:
    m = doc.get("metadata", {})
    return {
        "file_num": str(doc["file_num"]),
        "file_name": doc["file_name"],
        "doc_id": m.get("doc_id", ""),
        "version": str(m.get("version", "")),
        "effective_date": m.get("effective_date", ""),
        "scope": m.get("scope", "general"),
    }


def _name(num: str, by_num: dict[str, dict[str, Any]]) -> str:
    return by_num[num]["file_name"] if num in by_num else num


def resolve_active_policies(
    query_ctx: ExtractedQueryContext, documents: list[dict]
) -> PolicyResolutionResult:
    metas = {str(d["file_num"]): _meta(d) for d in documents}
    region = query_ctx.region

    ref = (
        query_ctx.contact_date
        or query_ctx.order_date
        or query_ctx.delivery_date
        or query_ctx.claim_date
    )
    # Section 3.2: promo-code validity is judged at the order date.
    promo_ref = query_ctx.order_date or query_ctx.contact_date or ref

    disclosures: list[MandatoryDisclosure] = []
    excluded: dict[str, ExcludedPolicyFile] = {}
    candidates: dict[str, dict[str, Any]] = {}

    def exclude(m: dict[str, Any], reason: str, detail: str) -> None:
        excluded[m["file_num"]] = ExcludedPolicyFile(**m, reason=reason, detail=detail)  # type: ignore[arg-type]

    if ref is None:
        disclosures.append(
            MandatoryDisclosure(
                kind="missing_date",
                files=[],
                message=(
                    "No date was given, so policy versions could not be matched to the date of "
                    "the customer's contact. Tell the customer which date you assumed or ask for it."
                ),
            )
        )

    for num in sorted(metas):
        m = metas[num]

        # Rule 4.1: regional addenda only apply to their own region.
        if m["scope"] == "regional" or num in REGIONAL_FILES:
            needed = REGIONAL_FILES.get(num)
            if needed != region:
                exclude(m, "regional_not_applicable",
                        f"Regional addendum for '{needed}'; customer region is '{region}'.")
                continue

        # Sections 3.1-3.2: in force only if effective_date <= relevant date.
        check_date = promo_ref if num == PROMO_FILE else ref
        if check_date is not None and m["effective_date"] > check_date:
            basis = "order date (Section 3.2)" if num == PROMO_FILE else "contact date (Section 3.1)"
            exclude(m, "not_yet_in_force",
                    f"Effective {m['effective_date']} is after the {basis} {check_date}.")
            continue

        candidates[num] = m

    # Rule 4.2: later effective_date supersedes within a topic group.
    for group, nums in SUPERSESSION_GROUPS.items():
        live = [candidates[n] for n in nums if n in candidates]
        if len(live) < 2:
            continue
        live.sort(key=lambda x: (x["effective_date"], x["file_num"]), reverse=True)
        winner = live[0]
        for loser in live[1:]:
            if loser["effective_date"] == winner["effective_date"]:
                continue
            candidates.pop(loser["file_num"], None)
            exclude(loser, "superseded",
                    f"Superseded by {winner['file_name']} (effective {winner['effective_date']}).")
            disclosures.append(
                MandatoryDisclosure(
                    kind="superseded_version",
                    files=[winner["file_name"], loser["file_name"]],
                    message=(
                        f"Two policy files cover the same topic and differ: {loser['file_name']} "
                        f"(effective {loser['effective_date']}) and {winner['file_name']} "
                        f"(effective {winner['effective_date']}). The later one, "
                        f"{winner['file_name']}, was used for this answer."
                    ),
                )
            )

    # Rule 4.1 disclosure: regional override of general/specific files.
    for num, needed in REGIONAL_FILES.items():
        if num in candidates and needed == region:
            disclosures.append(
                MandatoryDisclosure(
                    kind="regional_override",
                    files=[candidates[num]["file_name"]],
                    message=(
                        f"{candidates[num]['file_name']} is a regional addendum for '{region}' and "
                        f"overrides general and specific files on the topics it covers, whatever "
                        f"their dates. It applies here because the question concerns that region."
                    ),
                )
            )

    # Rule 4.4: same date, same scope, conflicting COD limits.
    a, b = COD_CONFLICT_FILES
    cod_relevant = (
        (query_ctx.payment_method or "").lower() in ("cod", "cash on delivery")
        or bool({"cod", "payments"} & {t.lower() for t in query_ctx.topics})
    )
    if (
        a in candidates and b in candidates and cod_relevant
        and ref is not None and ref >= COD_CONFLICT_START
        and candidates[a]["scope"] == candidates[b]["scope"]
        and candidates[a]["effective_date"] == candidates[b]["effective_date"]
    ):
        fa, fb = candidates[a]["file_name"], candidates[b]["file_name"]
        disclosures.append(
            MandatoryDisclosure(
                kind="unresolved_conflict",
                files=[fa, fb],
                message=(
                    f"{fa}, Section 3.2 sets the COD maximum total payable at Rs. 40,000, while "
                    f"{fb}, Section 1.2 sets it at Rs. 50,000. Both files are in force with the same "
                    f"date and scope. NovaCart support must confirm which limit applies."
                ),
            )
        )

    active = [
        ActivePolicyFile(
            **m,
            reason=("regional addendum for customer's region" if m["file_num"] in REGIONAL_FILES
                    else "in force on the relevant date"),
        )
        for _, m in sorted(candidates.items())
    ]
    return PolicyResolutionResult(
        reference_date=ref,
        promo_reference_date=promo_ref,
        region=region,
        active=active,
        excluded=[excluded[k] for k in sorted(excluded)],
        disclosures=disclosures,
    )
