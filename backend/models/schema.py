from __future__ import annotations
from typing import Optional, List, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from enum import Enum


class TrustTier(str, Enum):
    official = "official"
    directory = "directory"
    news = "news"
    unknown = "unknown"


class LocationMethod(str, Enum):
    string_match = "string_match"
    geocode = "geocode"
    llm_inference = "llm_inference"
    none = "none"


class Evidence(BaseModel):
    source_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    quote: str

    @model_validator(mode="after")
    def validate_offsets(self):
        if self.char_end < self.char_start:
            raise ValueError("char_end must be greater than or equal to char_start")
        return self


class GroundedString(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    abstain_reason: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_abstain_reason(self):
        if self.value is None and not self.abstain_reason:
            raise ValueError("abstain_reason is required when value is null")
        return self


class GroundedSiteType(GroundedString):
    value: Optional[Literal["Quarry"]] = None


class GroundedOperationalStatus(GroundedString):
    value: Optional[Literal["active", "inactive", "unknown"]] = None


class Source(BaseModel):
    source_id: str
    url: str
    fetched_at: str
    content_hash: str
    trust_tier: Optional[TrustTier] = None


class ReconciliationCandidate(BaseModel):
    value: Any = None
    source_id: str
    score: float


class Reconciliation(BaseModel):
    field: str
    candidates: List[ReconciliationCandidate]
    winner_source_id: str
    reason: str


class LocationVerification(BaseModel):
    is_verified: bool
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_city: Optional[str] = None
    method: LocationMethod


class Extraction(BaseModel):
    official_name: Optional[GroundedString] = None
    site_type: Optional[GroundedSiteType] = None
    description: Optional[GroundedString] = None
    materials_produced: List[GroundedString] = Field(default_factory=list)
    certifications: List[GroundedString] = Field(default_factory=list)
    operational_status: Optional[GroundedOperationalStatus] = None
    location_verification: Optional[LocationVerification] = None


class InputData(BaseModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    radius_km: float = Field(gt=0)


class ModelCall(BaseModel):
    model: str
    purpose: str
    tokens_in: int
    tokens_out: int
    usd_cost: float


class Metrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    llm_tokens_in: int
    llm_tokens_out: int
    usd_cost: float
    latency_ms: int
    model_calls: List[ModelCall] = Field(default_factory=list)


class RunMetadata(BaseModel):
    run_id: str
    prompt_hash: str
    scraper_version: str
    created_at: str


class Provenance(BaseModel):
    sources: List[Source]
    reconciliations: List[Reconciliation]


class QuarrySiteRecord(BaseModel):
    site_id: str
    schema_version: str = "2.0.0"
    input: InputData
    extraction: Extraction
    provenance: Provenance
    metrics: Metrics
    run_metadata: RunMetadata


class JobRequest(BaseModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    radius_km: float = Field(gt=0)
    max_usd_cost: Optional[float] = None


class JobResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    created_at: str
    updated_at: str
    result_count: int = 0
    error: Optional[str] = None
