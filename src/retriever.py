"""Step 2/3: Hybrid Lexical + Metadata Retrieval & Multi-Hop Cross-Reference Expansion."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schemas import (
    ExtractedQueryContext,
    PolicyResolutionResult,
    RetrievalResult,
    RetrievedChunk,
)

TOPIC_FILES: dict[str, list[str]] = {
    "returns": ["01", "02", "04"],
    "refunds": ["03", "01", "02"],
    "non_returnable": ["04"],
    "shipping": ["05", "06", "21"],
    "shipping_fees": ["06", "21"],
    "damaged_lost": ["07"],
    "warranty": ["08", "09"],
    "warranty_claims": ["09"],
    "payments": ["10", "11"],
    "cod": ["10", "11"],
    "promo": ["12"],
    "sales_bundles": ["13"],
    "store_credit": ["14"],
    "privacy": ["15", "16"],
    "data_requests": ["16"],
    "seller": ["17", "18"],
    "seller_returns": ["18"],
    "remote": ["19"],
    "international": ["20"],
    "checkout": ["00", "06", "12", "13"],
    "gst": ["00", "20"],
    "precedence": ["00"],
}

REGION_FILES: dict[str, str] = {"remote": "19", "international": "20"}
SPECIFIC_OVERRIDE_FILES: set[str] = {"13", "17", "18"}
SUPERSESSION_REDIRECTS: dict[str, str] = {
    "02": "01",
    "01": "02",
    "06": "21",
    "21": "06",
}

STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from how i if in is it me my of on or "
    "our so that the their this to was we what when where which who will with would you your".split()
)

PRESERVED_TOKENS = frozenset(
    {
        "cod",
        "gst",
        "freeship",
        "welcome10",
        "style15",
        "tech500",
        "ajk",
        "gb",
        "uae",
        "uk",
        "usa",
        "3-d",
        "rs",
    }
)

MAX_WHOLE_FILE_CHUNKS = 3
MAX_HOPS = 2
K1 = 1.5
B = 0.75


def load_policy_chunks(path: Path = Path("processed/policy_chunks.json")) -> list[dict]:
    """Load preprocessed policy chunks from JSON."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("chunks", [])
    return list(data)


def build_chunk_indices(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Index chunks by chunk_id, (file_num, section_num), (file_num, item_id), and file_num."""
    by_chunk_id: dict[str, dict[str, Any]] = {}
    by_section: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_item: dict[tuple[str, str], dict[str, Any]] = {}
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        cid = str(chunk.get("chunk_id", ""))
        fnum = str(chunk.get("file_num", ""))
        snum = str(chunk.get("section_num", ""))
        iid = chunk.get("item_id")

        if cid:
            by_chunk_id[cid] = chunk
        if fnum:
            by_file[fnum].append(chunk)
        if fnum and snum:
            by_section[(fnum, snum)].append(chunk)
        if fnum and iid is not None:
            by_item[(fnum, str(iid))] = chunk

    return {
        "by_chunk_id": by_chunk_id,
        "by_section": dict(by_section),
        "by_item": by_item,
        "by_file": dict(by_file),
    }


def tokenize(text: str) -> list[str]:
    """Normalize text while preserving domain tokens (3-d, 3.2, cod, gst, promo codes)."""
    if not text:
        return []
    lowered = text.lower()
    raw_tokens = re.findall(r"\d+\.\d+|3-d|[a-z0-9]+", lowered)
    out: list[str] = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue
        if tok not in PRESERVED_TOKENS and "." not in tok and "-" not in tok:
            if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
                tok = tok[:-1]
        out.append(tok)
    return out


def _chunk_tokens(chunk: dict[str, Any]) -> list[str]:
    title_block = f"{chunk.get('doc_title', '')} {chunk.get('section_title', '')}"
    return tokenize(chunk.get("text", "")) + tokenize(title_block) * 2


def _item_sort_key(chunk: dict[str, Any]) -> tuple[str, int, tuple[int, ...], str]:
    fnum = str(chunk.get("file_num", "99"))
    ctype = chunk.get("chunk_type", "item")
    type_rank = 0 if ctype == "overview" else (1 if ctype == "section_intro" else 2)
    iid = str(chunk.get("item_id") or "")
    parts = tuple(int(p) for p in iid.split(".") if p.isdigit()) if iid else (0,)
    return (fnum, type_rank, parts, str(chunk.get("chunk_id", "")))


def _resolve_target_chunks(
    target_file_num: str,
    target_section: str | None,
    indices: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve [file NN], [file NN, Section X], or [file NN, Section X.Y] against active chunks."""
    if not target_section:
        file_chunks = list(indices["by_file"].get(target_file_num, []))
        file_chunks.sort(key=_item_sort_key)
        # Prefer overview + first items, capped cleanly so whole-file refs don't flood context
        overviews = [c for c in file_chunks if c.get("chunk_type") in ("overview", "section_intro")]
        items = [c for c in file_chunks if c.get("chunk_type") == "item"]
        selected = (overviews[:1] + items[:MAX_WHOLE_FILE_CHUNKS])[:MAX_WHOLE_FILE_CHUNKS]
        return selected or file_chunks[:MAX_WHOLE_FILE_CHUNKS]

    if "." in target_section:
        item_chunk = indices["by_item"].get((target_file_num, target_section))
        if item_chunk is not None:
            return [item_chunk]
        # Fallback if a section has sub-items starting with target_section + "."
        sec_num = target_section.split(".", 1)[0]
        return [
            c
            for c in indices["by_section"].get((target_file_num, sec_num), [])
            if str(c.get("item_id") or "").startswith(target_section)
        ]

    sec_chunks = list(indices["by_section"].get((target_file_num, target_section), []))
    sec_chunks.sort(key=_item_sort_key)
    return sec_chunks


def _is_calculation_needed(query: str, query_ctx: ExtractedQueryContext) -> bool:
    if getattr(query_ctx, "is_calculation_required", False):
        return True
    if query_ctx.calc_type:
        return True
    if {"checkout", "gst"} & {t.lower() for t in query_ctx.topics}:
        return True
    q_low = (query or "").lower()
    return bool(re.search(r"\b(calculate|total payable|checkout|gst|round|how much)\b", q_low))


def _inject_foundational_conventions(
    primary_list: list[RetrievedChunk],
    seen_ids: set[str],
    active_indices: dict[str, Any],
    query: str,
    query_ctx: ExtractedQueryContext,
) -> None:
    """Ensure foundational conventions from File 00 are present when calculation/GST/rounding applies."""
    if "00" not in active_indices["by_file"]:
        return

    needed_items: list[str] = []
    q_low = (query or "").lower()
    topics_low = {t.lower() for t in query_ctx.topics}

    if _is_calculation_needed(query, query_ctx):
        needed_items.extend(["5.1", "1.3"])
    if "gst" in topics_low or "gst" in q_low or query_ctx.region == "international":
        needed_items.append("1.2")

    for iid in dict.fromkeys(needed_items):
        cand = active_indices["by_item"].get(("00", iid))
        if cand and cand["chunk_id"] not in seen_ids:
            seen_ids.add(cand["chunk_id"])
            primary_list.append(
                RetrievedChunk(
                    chunk=cand,
                    score=0.5,
                    retrieval_source="primary",
                    source="primary",
                )
            )


def retrieve_relevant_chunks(
    query: str,
    query_ctx: ExtractedQueryContext,
    resolution: PolicyResolutionResult,
    all_chunks: list[dict],
    top_k: int = 12,
) -> RetrievalResult:
    """Retrieve active chunks using BM25 + metadata boosts and expand cross-references up to 2 hops."""
    active_nums = resolution.active_file_nums
    excluded_reasons = {e.file_num: e.reason for e in resolution.excluded}

    # Strict Active Policy Filtering: exclude any chunk not in active_file_nums
    pool = [c for c in all_chunks if str(c.get("file_num")) in active_nums]
    if not pool:
        return RetrievalResult(query=query)

    active_indices = build_chunk_indices(pool)

    # Build BM25 statistics over the active corpus pool
    docs = [Counter(_chunk_tokens(c)) for c in pool]
    lengths = [sum(d.values()) for d in docs]
    avg_len = (sum(lengths) / len(lengths)) or 1.0

    df: Counter[str] = Counter()
    for d in docs:
        df.update(d.keys())
    n_docs = len(pool)

    # Combine raw query tokens with structured context signals
    extra_terms: list[str] = []
    extra_terms.extend(query_ctx.topics)
    extra_terms.extend(query_ctx.categories)
    if query_ctx.promo_code:
        extra_terms.append(query_ctx.promo_code)
    promo_list = getattr(query_ctx, "promo_codes_mentioned", None) or []
    extra_terms.extend(promo_list)
    if query_ctx.payment_method:
        extra_terms.append(query_ctx.payment_method)
    if query_ctx.region and query_ctx.region != "unspecified":
        extra_terms.append(query_ctx.region)

    q_terms = list(dict.fromkeys(tokenize(query) + tokenize(" ".join(extra_terms))))

    topic_files = {f for t in query_ctx.topics for f in TOPIC_FILES.get(t.lower(), [])}
    region_file = REGION_FILES.get(query_ctx.region)

    # Detect COD conflict pair relevance (Files 10 and 11 on/after 2026-05-01)
    cod_conflict_active = (
        {"10", "11"} <= active_nums
        and (
            (query_ctx.payment_method or "").lower() in ("cod", "cash on delivery")
            or bool({"cod", "payments"} & {t.lower() for t in query_ctx.topics})
            or bool(re.search(r"\b(cod|cash on delivery|order limit)\b", (query or "").lower()))
        )
    )

    context_overlap_tokens = set(
        tokenize(
            " ".join(
                list(query_ctx.topics)
                + list(query_ctx.categories)
                + ([query_ctx.promo_code] if query_ctx.promo_code else [])
                + list(promo_list)
            )
        )
    )

    scored: list[tuple[float, int]] = []
    for idx, (chunk, tf) in enumerate(zip(pool, docs)):
        fnum = str(chunk.get("file_num", ""))
        cid = str(chunk.get("chunk_id", ""))
        ctype = str(chunk.get("chunk_type", "item"))

        bm25 = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            bm25 += idf * (freq * (K1 + 1.0)) / (freq + K1 * (1.0 - B + B * lengths[idx] / avg_len))

        # Allow conflict-pair key clauses to surface even if lexical overlap is partial
        if bm25 <= 0.0 and not (cod_conflict_active and cid in ("10_3.2", "11_1.2")):
            continue

        score = max(bm25, 0.25)

        # 1. Title & structured topic/category/promo overlap boost
        title_tokens = set(tokenize(f"{chunk.get('doc_title', '')} {chunk.get('section_title', '')}"))
        q_title_hits = len(set(q_terms) & title_tokens)
        ctx_title_hits = len(context_overlap_tokens & title_tokens)
        score *= 1.0 + 0.12 * q_title_hits + 0.15 * ctx_title_hits

        # 2. Topic-mapped file boost
        if fnum in topic_files:
            score *= 1.35

        # 3. Regional override boost (Files 19 / 20)
        if region_file and fnum == region_file:
            score *= 1.55

        # 4. Specific override boost (Files 13, 17, 18) when seller/sales/bundles match
        if fnum in SPECIFIC_OVERRIDE_FILES:
            if fnum in ("17", "18") and (
                query_ctx.seller == "marketplace" or "seller" in {t.lower() for t in query_ctx.topics}
            ):
                score *= 1.45
            elif fnum == "13" and (
                "sales_bundles" in {t.lower() for t in query_ctx.topics}
                or bool(re.search(r"\b(sale|flash|bundle|final sale)\b", (query or "").lower()))
            ):
                score *= 1.45

        # 5. Unresolved conflict pair boost (10_3.2 and 11_1.2)
        if cod_conflict_active and fnum in ("10", "11"):
            score *= 1.4
            if cid in ("10_3.2", "11_1.2"):
                score *= 1.35

        # 6. Slight preference for numbered items over overviews/section intros on ties
        if ctype == "item":
            score *= 1.03

        scored.append((score, idx))

    scored.sort(key=lambda pair: (-pair[0], str(pool[pair[1]].get("chunk_id", ""))))

    primary_chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            chunk=pool[idx],
            score=round(sc, 4),
            retrieval_source="primary",
            source="primary",
        )
        for sc, idx in scored[:top_k]
    ]

    seen_ids: set[str] = {rc.chunk["chunk_id"] for rc in primary_chunks}
    _inject_foundational_conventions(primary_chunks, seen_ids, active_indices, query, query_ctx)

    # Multi-hop cross-reference expansion (up to MAX_HOPS=2) with cycle prevention
    expansion_trace: list[dict[str, Any]] = []
    expanded_map: dict[str, RetrievedChunk] = {}
    visited_edges: set[tuple[str, str, str]] = set()

    frontier: list[RetrievedChunk] = list(primary_chunks)
    for hop in range(1, MAX_HOPS + 1):
        next_frontier: list[RetrievedChunk] = []
        for parent in frontier:
            parent_cid = str(parent.chunk.get("chunk_id", ""))
            parent_citation = str(parent.chunk.get("citation") or parent_cid)

            for ref in parent.chunk.get("cross_references") or []:
                if ref.get("valid") is False:
                    continue

                raw_ref = str(ref.get("raw") or "")
                target_file = str(ref.get("target_file_num") or "")
                target_section = ref.get("target_section")
                target_section_str = str(target_section) if target_section is not None else None

                if not target_file:
                    continue

                edge_key = (parent_cid, target_file, target_section_str or "")
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                resolved_file = target_file
                action = "followed"
                reason = "target file is active"

                # Active-Policy Guard + Supersession Redirection
                if target_file not in active_nums:
                    excl_reason = excluded_reasons.get(target_file, "not_active")
                    redirect_candidate = SUPERSESSION_REDIRECTS.get(target_file)

                    if excl_reason == "superseded" and redirect_candidate in active_nums:
                        resolved_file = redirect_candidate
                        action = "redirected"
                        reason = f"target file {target_file} superseded by active file {resolved_file}"
                    else:
                        expansion_trace.append(
                            {
                                "hop": hop,
                                "from_chunk_id": parent_cid,
                                "from_citation": parent_citation,
                                "raw": raw_ref,
                                "target_file_num": target_file,
                                "target_section": target_section_str,
                                "resolved_file_num": None,
                                "action": "skipped",
                                "reason": f"target file not active ({excl_reason})",
                            }
                        )
                        continue

                target_candidates = _resolve_target_chunks(
                    resolved_file, target_section_str, active_indices
                )
                resolved_ids: list[str] = []

                for cand in target_candidates:
                    cand_id = str(cand["chunk_id"])
                    resolved_ids.append(cand_id)

                    if cand_id in seen_ids:
                        # If already added via cross-reference, append pointer provenance
                        if cand_id in expanded_map and parent_citation not in expanded_map[cand_id].referenced_by:
                            expanded_map[cand_id].referenced_by.append(parent_citation)
                        continue

                    seen_ids.add(cand_id)
                    rc = RetrievedChunk(
                        chunk=cand,
                        score=round(parent.score * (0.5 ** hop), 4),
                        retrieval_source="cross_reference",
                        source="cross_reference",
                        referenced_by=[parent_citation],
                        via_chunk_id=parent_cid,
                    )
                    expanded_map[cand_id] = rc
                    next_frontier.append(rc)

                expansion_trace.append(
                    {
                        "hop": hop,
                        "from_chunk_id": parent_cid,
                        "from_citation": parent_citation,
                        "raw": raw_ref,
                        "target_file_num": target_file,
                        "target_section": target_section_str,
                        "resolved_file_num": resolved_file,
                        "resolved_chunk_ids": resolved_ids,
                        "action": action,
                        "reason": reason,
                    }
                )

        frontier = next_frontier
        if not frontier:
            break

    # Sort cross-referenced chunks deterministically by (file_num, item_id)
    cross_referenced_chunks = sorted(
        expanded_map.values(),
        key=lambda rc: _item_sort_key(rc.chunk),
    )

    all_retrieved = [*primary_chunks, *cross_referenced_chunks]
    skipped_refs = [t for t in expansion_trace if t["action"] == "skipped"]

    return RetrievalResult(
        query=query,
        primary_chunks=primary_chunks,
        cross_referenced_chunks=cross_referenced_chunks,
        all_chunks=all_retrieved,
        chunks=all_retrieved,
        expansion_trace=expansion_trace,
        skipped_cross_references=skipped_refs,
    )