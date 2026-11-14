"""Pydantic v2 models shared across the NovaCart pipeline."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Region = Literal["metro", "remote", "international", "rest_of_pakistan", "unspecified"]
Seller = Literal["novacart", "marketplace", "unspecified"]
DisclosureKind = Literal[
    "superseded_version",
    "unresolved_conflict",
    "regional_override",
    "missing_date",
]
ExclusionReason = Literal["not_yet_in_force", "superseded", "regional_not_applicable"]


def _valid_iso_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        return None


class ExtractedQueryContext(BaseModel):
    """Structured facts pulled from the customer's message (Step 1)."""

    user_query: str
    contact_date: str | None = None
    order_date: str | None = None
    delivery_date: str | None = None
    claim_date: str | None = None
    destination: str | None = None
    region: Region = "unspecified"
    categories: list[str] = Field(default_factory=list)
    seller: Seller = "unspecified"
    topics: list[str] = Field(default_factory=list)
    promo_code: str | None = None
    payment_method: str | None = None
    order_total: int | None = None
    calc_type: str | None = None
    calc_inputs: dict[str, Any] = Field(default_factory=dict)
    injection_detected: bool = False
    extraction_source: Literal["llm", "heuristic"] = "heuristic"

    @field_validator("contact_date", "order_date", "delivery_date", "claim_date", mode="before")
    @classmethod
    def _check_dates(cls, v: Any) -> str | None:
        return _valid_iso_or_none(v)

    @field_validator("promo_code", mode="before")
    @classmethod
    def _upper_promo(cls, v: Any) -> str | None:
        if v is None or str(v).strip() == "":
            return None
        return str(v).strip().upper()

    @property
    def is_calculation_required(self) -> bool:
        """Whether the query asks for deterministic arithmetic."""
        calculation_topics = {"checkout", "gst", "refunds", "warranty"}
        return bool(
            self.calc_type
            or calculation_topics.intersection(topic.lower() for topic in self.topics)
        )


class ActivePolicyFile(BaseModel):
    file_num: str
    file_name: str
    doc_id: str
    version: str
    effective_date: str
    scope: str
    reason: str = ""


class ExcludedPolicyFile(BaseModel):
    file_num: str
    file_name: str
    doc_id: str
    version: str
    effective_date: str
    scope: str
    reason: ExclusionReason
    detail: str = ""


class MandatoryDisclosure(BaseModel):
    """Something the final answer MUST tell the customer."""

    kind: DisclosureKind
    files: list[str]
    message: str


class PolicyResolutionResult(BaseModel):
    reference_date: str | None
    promo_reference_date: str | None = None
    region: Region = "unspecified"
    active: list[ActivePolicyFile] = Field(default_factory=list)
    excluded: list[ExcludedPolicyFile] = Field(default_factory=list)
    disclosures: list[MandatoryDisclosure] = Field(default_factory=list)

    @property
    def active_file_nums(self) -> set[str]:
        return {a.file_num for a in self.active}

    @property
    def active_file_names(self) -> set[str]:
        return {a.file_name for a in self.active}

    @property
    def mandatory_disclosures(self) -> list[MandatoryDisclosure]:
        return self.disclosures


class RetrievedChunk(BaseModel):
    chunk: dict[str, Any]
    score: float = 0.0
    retrieval_source: Literal["primary", "cross_reference"] = "primary"
    source: Literal["direct", "primary", "cross_reference"] = "primary"
    referenced_by: list[str] = Field(default_factory=list)
    via_chunk_id: str | None = None

    @model_validator(mode="after")
    def _sync_source_fields(self) -> "RetrievedChunk":
        if self.source == "cross_reference" or self.retrieval_source == "cross_reference":
            self.retrieval_source = "cross_reference"
            self.source = "cross_reference"
        elif self.source == "direct":
            self.retrieval_source = "primary"
        else:
            self.retrieval_source = "primary"
            self.source = "primary"
        return self


class RetrievalResult(BaseModel):
    query: str
    primary_chunks: list[RetrievedChunk] = Field(default_factory=list)
    cross_referenced_chunks: list[RetrievedChunk] = Field(default_factory=list)
    all_chunks: list[RetrievedChunk] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    expansion_trace: list[dict[str, Any]] = Field(default_factory=list)
    skipped_cross_references: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sync_chunk_lists(self) -> "RetrievalResult":
        if not self.all_chunks and (self.primary_chunks or self.cross_referenced_chunks):
            self.all_chunks = [*self.primary_chunks, *self.cross_referenced_chunks]
        elif self.all_chunks and not (self.primary_chunks or self.cross_referenced_chunks):
            self.primary_chunks = [c for c in self.all_chunks if c.retrieval_source == "primary"]
            self.cross_referenced_chunks = [
                c for c in self.all_chunks if c.retrieval_source == "cross_reference"
            ]
        elif not self.all_chunks and self.chunks:
            self.all_chunks = list(self.chunks)
            self.primary_chunks = [c for c in self.all_chunks if c.retrieval_source == "primary"]
            self.cross_referenced_chunks = [
                c for c in self.all_chunks if c.retrieval_source == "cross_reference"
            ]

        self.chunks = list(self.all_chunks)
        if not self.skipped_cross_references and self.expansion_trace:
            self.skipped_cross_references = [
                t for t in self.expansion_trace if t.get("action") == "skipped"
            ]
        return self

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk["chunk_id"] for c in self.all_chunks]


class CalculationTrace(BaseModel):
    """One deterministic calculation result and its policy citations."""

    name: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)


class BotResponse(BaseModel):
    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    disclosures: list[MandatoryDisclosure] = Field(default_factory=list)
    context: ExtractedQueryContext
    resolution: PolicyResolutionResult
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    calculations: dict[str, Any] = Field(default_factory=dict)
    calculation_trace: list[CalculationTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    injection_detected: bool = False

    @property
    def citations_used(self) -> list[str]:
        return self.citations

    @property
    def active_policies(self) -> list[ActivePolicyFile]:
        return self.resolution.active

    @property
    def mandatory_disclosures(self) -> list[MandatoryDisclosure]:
        return self.disclosures

    @property
    def query_context(self) -> ExtractedQueryContext:
        return self.context

    @property
    def injection_flagged(self) -> bool:
        return self.injection_detected


QueryContext = ExtractedQueryContext