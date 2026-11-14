"""Comprehensive unit tests for Step 2 Hybrid Retrieval & Cross-Reference Expansion."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policy_resolver import load_policy_documents, resolve_active_policies  # noqa: E402
from src.query_extractor import normalize_region  # noqa: E402
from src.retriever import load_policy_chunks, retrieve_relevant_chunks  # noqa: E402
from src.schemas import ExtractedQueryContext  # noqa: E402

DOCS_PATH = Path("processed/policy_documents.json")
CHUNKS_PATH = Path("processed/policy_chunks.json")
POLICIES_DIR = Path("policies")


@pytest.fixture(scope="module")
def corpus_documents() -> list[dict]:
    return load_policy_documents(DOCS_PATH)


@pytest.fixture(scope="module")
def corpus_chunks() -> list[dict]:
    return load_policy_chunks(CHUNKS_PATH)


def _ctx(**kwargs) -> ExtractedQueryContext:
    kwargs.setdefault("user_query", "policy question")
    return ExtractedQueryContext(**kwargs)


def _hash_policies_dir() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for md_file in sorted(POLICIES_DIR.glob("*.md")):
        hashes[md_file.name] = hashlib.sha256(md_file.read_bytes()).hexdigest()
    return hashes


# ---------------------------------------------------------------------------
# 1. Strict Active-File Isolation
# ---------------------------------------------------------------------------


def test_2026_shipping_query_excludes_2027_file(corpus_documents, corpus_chunks):
    ctx = _ctx(
        user_query="What are the standard shipping fees and free shipping threshold?",
        contact_date="2026-06-15",
        region="metro",
        topics=["shipping_fees", "shipping"],
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=10)

    assert len(result.primary_chunks) > 0
    assert all(rc.chunk["file_num"] != "21" for rc in result.all_chunks)
    assert any(rc.chunk["file_num"] == "06" for rc in result.primary_chunks)


def test_2025_returns_query_uses_file_02_and_excludes_file_01(corpus_documents, corpus_chunks):
    ctx = _ctx(
        user_query="What is the return window and restocking fee for returning an item?",
        contact_date="2025-11-15",
        region="metro",
        topics=["returns", "refunds"],
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=10)

    file_nums = {rc.chunk["file_num"] for rc in result.all_chunks}
    assert "02" in file_nums
    assert "01" not in file_nums


# ---------------------------------------------------------------------------
# 2. 1-Hop Item-Level Cross-Reference Expansion
# ---------------------------------------------------------------------------


def test_one_hop_item_level_expansion(corpus_documents, corpus_chunks):
    # Find an active chunk in the corpus that contains a dotted item-level xref [file NN, Section X.Y]
    ctx = _ctx(
        user_query="How are promo code discounts calculated and split on eligible items?",
        contact_date="2026-06-15",
        order_date="2026-06-15",
        region="metro",
        topics=["promo", "checkout"],
        promo_code="STYLE15",
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=10)

    item_traces = [
        t
        for t in result.expansion_trace
        if t["action"] == "followed" and t.get("target_section") and "." in t["target_section"]
    ]
    assert len(item_traces) >= 1

    target_cid = item_traces[0]["resolved_chunk_ids"][0]
    xref_chunk = next(
        (rc for rc in result.cross_referenced_chunks if rc.chunk["chunk_id"] == target_cid),
        None,
    )
    assert xref_chunk is not None
    assert xref_chunk.retrieval_source == "cross_reference"
    assert len(xref_chunk.referenced_by) >= 1
    assert ", Section " in xref_chunk.referenced_by[0]


# ---------------------------------------------------------------------------
# 3. 1-Hop Section-Level Cross-Reference Expansion
# ---------------------------------------------------------------------------


def test_one_hop_section_level_expansion(corpus_documents, corpus_chunks):
    # File 06 Section 2.2 references [file 00, Section 5] (checkout calculation order)
    ctx = _ctx(
        user_query="How is the standard shipping fee applied after discounts?",
        contact_date="2026-06-15",
        region="metro",
        topics=["shipping_fees"],
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=8)

    all_ids = set(result.chunk_ids)
    # Section 5 of File 00 has items 5.1, 5.2, 5.3
    assert {"00_5.1", "00_5.2", "00_5.3"} <= all_ids

    sec_traces = [
        t
        for t in result.expansion_trace
        if t["action"] == "followed"
        and t["target_file_num"] == "00"
        and t["target_section"] == "5"
    ]
    assert len(sec_traces) >= 1


# ---------------------------------------------------------------------------
# 4. Regional Override Retrieval (Files 19 & 20)
# ---------------------------------------------------------------------------


def test_regional_override_remote_gilgit_baltistan(corpus_documents, corpus_chunks):
    region = normalize_region("Gilgit-Baltistan", "unspecified")
    ctx = _ctx(
        user_query="What are the shipping fees and delivery rules for Gilgit-Baltistan?",
        contact_date="2026-06-15",
        destination="Gilgit-Baltistan",
        region=region,
        topics=["shipping_fees", "remote"],
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=8)

    top_files = [rc.chunk["file_num"] for rc in result.primary_chunks[:4]]
    assert "19" in top_files
    assert all(rc.chunk["file_num"] != "20" for rc in result.all_chunks)


def test_regional_override_international_uae_uk(corpus_documents, corpus_chunks):
    region = normalize_region("Dubai, UAE", "unspecified")
    ctx = _ctx(
        user_query="Do international orders to the UAE include GST and can they be returned?",
        contact_date="2026-06-15",
        destination="Dubai, UAE",
        region=region,
        topics=["international", "gst", "returns"],
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=8)

    top_files = [rc.chunk["file_num"] for rc in result.primary_chunks[:4]]
    assert "20" in top_files
    assert all(rc.chunk["file_num"] != "19" for rc in result.all_chunks)


def test_metro_lahore_excludes_regional_addenda(corpus_documents, corpus_chunks):
    region = normalize_region("Lahore", "unspecified")
    ctx = _ctx(
        user_query="What are the shipping fees and return rules for Lahore?",
        contact_date="2026-06-15",
        destination="Lahore",
        region=region,
        topics=["shipping_fees", "returns"],
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=10)

    retrieved_files = {rc.chunk["file_num"] for rc in result.all_chunks}
    assert "19" not in retrieved_files
    assert "20" not in retrieved_files


# ---------------------------------------------------------------------------
# 5. Unresolved Conflict Retrieval (File 10 vs File 11 COD Limit)
# ---------------------------------------------------------------------------


def test_unresolved_cod_conflict_retrieves_both_10_and_11(corpus_documents, corpus_chunks):
    ctx = _ctx(
        user_query="Can I pay Rs. 45,000 using cash on delivery (COD)?",
        contact_date="2026-05-10",
        region="metro",
        payment_method="cod",
        order_total=45000,
        topics=["cod", "payments"],
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=8)

    primary_ids = {rc.chunk["chunk_id"] for rc in result.primary_chunks}
    assert "10_3.2" in primary_ids
    assert "11_1.2" in primary_ids


# ---------------------------------------------------------------------------
# 6. Calculation Order Injection
# ---------------------------------------------------------------------------


def test_calculation_order_injected_when_calculation_required(corpus_documents, corpus_chunks):
    ctx = _ctx(
        user_query="Calculate total payable for a shirt with WELCOME10.",
        contact_date="2026-06-15",
        order_date="2026-06-15",
        region="metro",
        topics=["promo"],
        promo_code="WELCOME10",
        calc_type="checkout",
    )
    resolution = resolve_active_policies(ctx, corpus_documents)
    result = retrieve_relevant_chunks(ctx.user_query, ctx, resolution, corpus_chunks, top_k=5)

    assert "00_5.1" in set(result.chunk_ids)


# ---------------------------------------------------------------------------
# 7. Source Integrity Verification
# ---------------------------------------------------------------------------


def test_policies_directory_untouched(corpus_documents):
    before_hashes = _hash_policies_dir()
    assert len(before_hashes) == 22

    # Compare against SHA-256 digests recorded during Step 0 preprocessing
    recorded_hashes = {
        doc["file_name"]: doc["sha256"]
        for doc in corpus_documents
        if "sha256" in doc
    }
    assert before_hashes == recorded_hashes