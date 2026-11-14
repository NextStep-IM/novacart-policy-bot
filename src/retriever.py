"""Step 3: lexical retrieval (BM25) over chunks of the active policy files."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
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
REGION_FILES = {"remote": "19", "international": "20"}
STOPWORDS = frozenset(
    "a an and are as at be by can could do does for from how i if in is it me my of on or "
    "our so that the their this to was we what when where which who will with would you your".split()
)
MAX_EXPANSION_PER_REF = 3
K1, B = 1.5, 0.75


def load_policy_chunks(path: Path = Path("processed/policy_chunks.json")) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("chunks", [])
    return list(data)


def _tokens(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def _chunk_tokens(chunk: dict[str, Any]) -> list[str]:
    title = f"{chunk.get('doc_title', '')} {chunk.get('section_title', '')}"
    return _tokens(chunk.get("text", "")) + _tokens(title) * 2


def _matches_reference(chunk: dict[str, Any], target_section: str) -> bool:
    item_id = str(chunk.get("item_id") or "")
    section = str(chunk.get("section_num") or "")
    return item_id == target_section or section == target_section or item_id.startswith(target_section + ".")


def retrieve_relevant_chunks(
    query: str,
    query_ctx: ExtractedQueryContext,
    resolution: PolicyResolutionResult,
    all_chunks: list[dict],
    top_k: int = 12,
) -> RetrievalResult:
    active = resolution.active_file_nums
    pool = [c for c in all_chunks if str(c.get("file_num")) in active]
    if not pool:
        return RetrievalResult(query=query)

    docs = [Counter(_chunk_tokens(c)) for c in pool]
    lengths = [sum(d.values()) for d in docs]
    avg_len = (sum(lengths) / len(lengths)) or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(pool)

    q_terms = list(dict.fromkeys(_tokens(query)))
    topic_files = {f for t in query_ctx.topics for f in TOPIC_FILES.get(t, [])}
    region_file = REGION_FILES.get(query_ctx.region)

    scored: list[tuple[float, int]] = []
    for i, (chunk, tf) in enumerate(zip(pool, docs)):
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * lengths[i] / avg_len))
        if score <= 0:
            continue
        title_hits = len(set(q_terms) & set(_tokens(chunk.get("section_title", ""))))
        score *= 1 + 0.15 * title_hits
        if str(chunk.get("file_num")) in topic_files:
            score *= 1.4
        if region_file and str(chunk.get("file_num")) == region_file:
            score *= 1.3
        scored.append((score, i))

    scored.sort(key=lambda s: (-s[0], str(pool[s[1]].get("chunk_id"))))
    direct = [RetrievedChunk(chunk=pool[i], score=round(s, 4)) for s, i in scored[:top_k]]

    # 1-hop cross-reference expansion, restricted to active files.
    seen = {c.chunk["chunk_id"] for c in direct}
    expanded: list[RetrievedChunk] = []
    skipped: list[dict[str, Any]] = []
    for item in direct:
        for ref in item.chunk.get("cross_references") or []:
            if ref.get("valid") is False:
                continue
            target_file = str(ref.get("target_file_num") or "")
            target_section = str(ref.get("target_section") or "")
            if not target_file:
                continue
            if target_file not in active:
                skipped.append({
                    "from_chunk_id": item.chunk["chunk_id"],
                    "raw": ref.get("raw"),
                    "target_file_num": target_file,
                    "reason": "target file not active for this question",
                })
                continue
            added = 0
            for cand in pool:
                if str(cand.get("file_num")) != target_file:
                    continue
                if target_section and not _matches_reference(cand, target_section):
                    continue
                if cand["chunk_id"] in seen:
                    continue
                seen.add(cand["chunk_id"])
                expanded.append(RetrievedChunk(
                    chunk=cand,
                    score=round(item.score * 0.5, 4),
                    source="cross_reference",
                    via_chunk_id=item.chunk["chunk_id"],
                ))
                added += 1
                if added >= MAX_EXPANSION_PER_REF:
                    break

    return RetrievalResult(query=query, chunks=direct + expanded, skipped_cross_references=skipped)
