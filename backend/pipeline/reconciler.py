# Merges extractions from multiple sources into one final QuarrySiteRecord.
# Picks the best value per field using trust tier + confidence scoring.
from __future__ import annotations
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from models.schema import (
    Evidence, Extraction, GroundedOperationalStatus, GroundedSiteType,
    GroundedString, InputData, LocationVerification, LocationMethod,
    Metrics, ModelCall, Provenance, QuarrySiteRecord, Reconciliation,
    ReconciliationCandidate, RunMetadata, Source,
)
from pipeline.extractor import EXTRACTION_PROMPT, EXTRACTION_SYSTEM, GEMINI_MODEL, SCRAPER_VERSION

# trust order: official > directory > news > unknown
TRUST_ORDER = {"official": 3, "directory": 2, "news": 1, "unknown": 0}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stable_site_id(osm_id: Optional[str], lat: float, lon: float, candidate_name: str) -> str:
    normalized_name = " ".join(candidate_name.lower().split()) if candidate_name else ""
    raw = f"{osm_id or 'no-osm'}:{lat:.6f}:{lon:.6f}:{normalized_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _prompt_hash_short() -> str:
    prompt_signature = "\n".join([
        GEMINI_MODEL,
        SCRAPER_VERSION,
        EXTRACTION_SYSTEM.strip(),
        EXTRACTION_PROMPT.strip(),
    ])
    return "sha256:" + hashlib.sha256(prompt_signature.encode()).hexdigest()[:16]


def _pick_best_grounded(
    field_name: str,
    candidates: List[Tuple[Dict, str, str]],  # (grounded_dict, source_id, trust_tier)
) -> Tuple[Dict, List[ReconciliationCandidate], str, str]:
    # score each candidate, highest trust tier wins, confidence breaks ties
    scored: List[Tuple[float, Dict, str, str]] = []
    for gd, src_id, tier in candidates:
        base_score = TRUST_ORDER.get(tier, 0) * 0.4
        conf = gd.get("confidence", 0.0)
        value_bonus = 0.2 if gd.get("value") is not None else 0.0
        total = base_score + conf * 0.4 + value_bonus
        scored.append((total, gd, src_id, tier))

    scored.sort(key=lambda x: -x[0])

    recon_candidates = [
        ReconciliationCandidate(value=gd.get("value"), source_id=src, score=round(score, 3))
        for score, gd, src, tier in scored
    ]

    if not scored:
        empty = {"value": None, "confidence": 0.0, "abstain_reason": "not_found", "evidence": []}
        return empty, [], "none", "no candidates"

    best_score, best_gd, best_src, best_tier = scored[0]
    reason = f"Highest combined score ({best_score:.2f}): trust_tier={best_tier}, confidence={best_gd.get('confidence', 0):.2f}"
    if len(scored) > 1:
        alt_score, _, alt_src, alt_tier = scored[1]
        reason += f"; runner-up {alt_src} (score={alt_score:.2f}, tier={alt_tier})"

    return best_gd, recon_candidates, best_src, reason


def _grounded_dict_to_model(gd: Dict, model_cls=GroundedString) -> GroundedString:
    evidence_models = [
        Evidence(
            source_id=e["source_id"],
            char_start=e["char_start"],
            char_end=e["char_end"],
            quote=e["quote"],
        )
        for e in gd.get("evidence", [])
    ]
    return model_cls(
        value=gd.get("value"),
        confidence=max(0.0, min(1.0, gd.get("confidence", 0.0))),
        abstain_reason=gd.get("abstain_reason"),
        evidence=evidence_models,
    )


def _verify_location(
    location_hints: List[Optional[str]],
    expected_city: Optional[str],
) -> LocationVerification:
    # expected_city is resolved once per job by the caller, not per candidate
    non_null_hints = [h for h in location_hints if h]

    if not expected_city and not non_null_hints:
        return LocationVerification(
            is_verified=False, confidence=0.0, extracted_city=None, method=LocationMethod.none
        )

    if non_null_hints and expected_city:
        expected_lower = expected_city.lower()
        for hint in non_null_hints:
            if expected_lower in hint.lower() or hint.lower() in expected_lower:
                return LocationVerification(
                    is_verified=True,
                    confidence=0.80,
                    extracted_city=hint,
                    method=LocationMethod.string_match,
                )
        return LocationVerification(
            is_verified=False,
            confidence=0.30,
            extracted_city=non_null_hints[0],
            method=LocationMethod.string_match,
        )

    if non_null_hints:
        return LocationVerification(
            is_verified=False,
            confidence=0.20,
            extracted_city=non_null_hints[0],
            method=LocationMethod.llm_inference,
        )

    return LocationVerification(
        is_verified=False, confidence=0.10, extracted_city=None, method=LocationMethod.geocode
    )


def reconcile(
    candidate,
    scraped_extractions: List[Tuple[Dict, str, str, str]],  # (extraction, source_id, url, tier)
    osm_extraction: Dict,
    osm_source_id: str,
    model_calls: List[Dict],
    input_data: dict,
    latency_ms: int,
    fetched_at: str,
    source_content_hashes: Dict[str, str],
    expected_city: Optional[str] = None,
) -> QuarrySiteRecord:

    run_id = str(uuid.uuid4())[:8]
    now = _now_iso()

    # build the source list - OSM first, then scraped sources
    sources: List[Source] = []
    osm_content_hash = source_content_hashes.get(osm_source_id, "sha256:osm")
    sources.append(Source(
        source_id=osm_source_id,
        url=f"https://www.openstreetmap.org/{candidate.osm_id or 'node/0'}",
        fetched_at=fetched_at,
        content_hash=osm_content_hash,
        trust_tier="official",
    ))

    for _ex, src_id, url, tier in scraped_extractions:
        sources.append(Source(
            source_id=src_id,
            url=url,
            fetched_at=fetched_at,
            content_hash=source_content_hashes.get(src_id, "sha256:unknown"),
            trust_tier=tier,
        ))

    all_extractions: List[Tuple[Dict, str, str]] = [(osm_extraction, osm_source_id, "official")]
    for ex, src_id, _url, tier in scraped_extractions:
        all_extractions.append((ex, src_id, tier))

    reconciliations: List[Reconciliation] = []
    location_hints: List[Optional[str]] = []

    def _reconcile_field(field_name: str, model_cls=GroundedString) -> GroundedString:
        cands = [(ex[field_name], src, tier) for ex, src, tier in all_extractions if field_name in ex]
        winner, recon_cands, winner_src, reason = _pick_best_grounded(field_name, cands)
        if len(recon_cands) > 1:
            reconciliations.append(Reconciliation(
                field=field_name,
                candidates=recon_cands,
                winner_source_id=winner_src,
                reason=reason,
            ))
        return _grounded_dict_to_model(winner, model_cls=model_cls)

    official_name = _reconcile_field("official_name")
    site_type = _reconcile_field("site_type", model_cls=GroundedSiteType)
    description = _reconcile_field("description")
    operational_status = _reconcile_field("operational_status", model_cls=GroundedOperationalStatus)

    for ex, _src, _tier in all_extractions:
        location_hints.append(ex.get("location_hint"))

    # union of all materials across sources, deduped by lowercase value
    all_materials = []
    seen_mats = set()
    for ex, src, tier in all_extractions:
        for m in (ex.get("materials_produced") or []):
            val = m.get("value")
            if val and val.lower() not in seen_mats:
                seen_mats.add(val.lower())
                all_materials.append(_grounded_dict_to_model(m))

    all_certs = []
    seen_certs = set()
    for ex, src, tier in all_extractions:
        for c in (ex.get("certifications") or []):
            val = c.get("value")
            if val and val.lower() not in seen_certs:
                seen_certs.add(val.lower())
                all_certs.append(_grounded_dict_to_model(c))

    location_verification = _verify_location(location_hints, expected_city)

    extraction = Extraction(
        official_name=official_name,
        site_type=site_type,
        description=description,
        materials_produced=all_materials,
        certifications=all_certs,
        operational_status=operational_status,
        location_verification=location_verification,
    )

    total_in = sum(mc.get("tokens_in", 0) for mc in model_calls)
    total_out = sum(mc.get("tokens_out", 0) for mc in model_calls)
    total_cost = sum(mc.get("usd_cost", 0.0) for mc in model_calls)
    mc_models = [
        ModelCall(
            model=mc["model"],
            purpose=mc["purpose"],
            tokens_in=mc["tokens_in"],
            tokens_out=mc["tokens_out"],
            usd_cost=mc["usd_cost"],
        )
        for mc in model_calls
    ]

    metrics = Metrics(
        llm_tokens_in=total_in,
        llm_tokens_out=total_out,
        usd_cost=total_cost,
        latency_ms=latency_ms,
        model_calls=mc_models,
    )

    run_metadata = RunMetadata(
        run_id=run_id,
        prompt_hash=_prompt_hash_short(),
        scraper_version=SCRAPER_VERSION,
        created_at=now,
    )

    site_id = _stable_site_id(candidate.osm_id, candidate.latitude, candidate.longitude, candidate.name)

    return QuarrySiteRecord(
        site_id=site_id,
        schema_version="2.0.0",
        input=InputData(
            latitude=input_data["latitude"],
            longitude=input_data["longitude"],
            radius_km=input_data["radius_km"],
        ),
        extraction=extraction,
        provenance=Provenance(sources=sources, reconciliations=reconciliations),
        metrics=metrics,
        run_metadata=run_metadata,
    )
