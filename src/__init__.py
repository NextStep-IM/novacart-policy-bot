"""NovaCart Policy Q&A Bot (Tech Fusion '26 - AI Hackathon - Challenge A)."""
from .retriever import build_chunk_indices, load_policy_chunks, retrieve_relevant_chunks
from .schemas import (
    ActivePolicyFile,
    BotResponse,
    ExcludedPolicyFile,
    ExtractedQueryContext,
    MandatoryDisclosure,
    PolicyResolutionResult,
    RetrievalResult,
    RetrievedChunk,
)
from .query_extractor import (
    build_default_llm_callable,
    contains_injection,
    extract_query_context,
    normalize_region,
    strip_injection_lines,
)
from .policy_resolver import load_policy_documents, resolve_active_policies
from .retriever import load_policy_chunks, retrieve_relevant_chunks
from .calculator import (
    add_business_days,
    add_calendar_days,
    calculate_checkout_total,
    calculate_prorated_warranty_refund,
    calculate_return_refund,
    compute_international_export_price,
    count_full_months,
    extract_gst_from_inclusive,
    round_half_up,
    split_amount_proportionally,
)
from .bot import NovaCartPolicyBot

__all__ = [
    "ActivePolicyFile",
    "BotResponse",
    "ExcludedPolicyFile",
    "ExtractedQueryContext",
    "MandatoryDisclosure",
    "NovaCartPolicyBot",
    "PolicyResolutionResult",
    "RetrievalResult",
    "RetrievedChunk",
    "add_business_days",
    "add_calendar_days",
    "build_default_llm_callable",
    "calculate_checkout_total",
    "calculate_prorated_warranty_refund",
    "calculate_return_refund",
    "compute_international_export_price",
    "contains_injection",
    "count_full_months",
    "extract_gst_from_inclusive",
    "extract_query_context",
    "load_policy_chunks",
    "load_policy_documents",
    "normalize_region",
    "resolve_active_policies",
    "retrieve_relevant_chunks",
    "round_half_up",
    "split_amount_proportionally",
    "strip_injection_lines",
]
