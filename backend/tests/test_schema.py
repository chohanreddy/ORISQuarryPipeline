# Schema validation tests - confidence ranges, required fields, trust tiers, abstain behaviour
import pytest
from pydantic import ValidationError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.schema import (
    Evidence, GroundedOperationalStatus, GroundedSiteType, GroundedString,
    InputData, Extraction, Provenance, Metrics, QuarrySiteRecord,
    RunMetadata, Source, LocationVerification, LocationMethod,
)


def _minimal_record(**overrides) -> dict:
    base = {
        "site_id": "abc123",
        "schema_version": "2.0.0",
        "input": {"latitude": 48.8566, "longitude": 2.3522, "radius_km": 50.0},
        "extraction": {},
        "provenance": {"sources": [], "reconciliations": []},
        "metrics": {
            "llm_tokens_in": 100,
            "llm_tokens_out": 50,
            "usd_cost": 0.001,
            "latency_ms": 1000,
            "model_calls": [],
        },
        "run_metadata": {
            "run_id": "r_001",
            "prompt_hash": "sha256:abc",
            "scraper_version": "1.0.0",
            "created_at": "2025-01-01T00:00:00Z",
        },
    }
    base.update(overrides)
    return base


class TestGroundedString:
    def test_valid_with_value(self):
        gs = GroundedString(
            value="Carrière de Test",
            confidence=0.9,
            evidence=[Evidence(source_id="src_1", char_start=10, char_end=28, quote="Carrière de Test")],
        )
        assert gs.value == "Carrière de Test"
        assert gs.confidence == 0.9

    def test_null_value_requires_abstain_reason(self):
        # pipeline must set abstain_reason when value is null
        gs = GroundedString(value=None, confidence=0.0, abstain_reason="not_found", evidence=[])
        assert gs.value is None
        assert gs.abstain_reason == "not_found"

    def test_null_value_without_abstain_reason_raises(self):
        with pytest.raises(ValidationError):
            GroundedString(value=None, confidence=0.0, evidence=[])

    def test_confidence_clamped_below_zero_raises(self):
        with pytest.raises(ValidationError):
            GroundedString(value="x", confidence=-0.1, evidence=[])

    def test_confidence_clamped_above_one_raises(self):
        with pytest.raises(ValidationError):
            GroundedString(value="x", confidence=1.1, evidence=[])

    def test_confidence_boundary_values(self):
        gs_low = GroundedString(value=None, confidence=0.0, abstain_reason="not_found", evidence=[])
        gs_high = GroundedString(value="x", confidence=1.0, evidence=[])
        assert gs_low.confidence == 0.0
        assert gs_high.confidence == 1.0

    def test_evidence_char_start_negative_raises(self):
        with pytest.raises(ValidationError):
            Evidence(source_id="s", char_start=-1, char_end=5, quote="test")

    def test_evidence_char_end_before_start_raises(self):
        with pytest.raises(ValidationError):
            Evidence(source_id="s", char_start=5, char_end=4, quote="test")

    def test_valid_full_record(self):
        record = QuarrySiteRecord(**_minimal_record())
        assert record.schema_version == "2.0.0"
        assert record.site_id == "abc123"

    def test_missing_required_field_raises(self):
        data = _minimal_record()
        del data["site_id"]
        with pytest.raises(ValidationError):
            QuarrySiteRecord(**data)

    def test_missing_metrics_raises(self):
        data = _minimal_record()
        del data["metrics"]
        with pytest.raises(ValidationError):
            QuarrySiteRecord(**data)

    def test_latitude_out_of_range(self):
        with pytest.raises(ValidationError):
            InputData(latitude=91.0, longitude=0.0, radius_km=10.0)

    def test_longitude_out_of_range(self):
        with pytest.raises(ValidationError):
            InputData(latitude=0.0, longitude=181.0, radius_km=10.0)

    def test_radius_km_must_be_positive(self):
        with pytest.raises(ValidationError):
            InputData(latitude=0.0, longitude=0.0, radius_km=0.0)

    def test_radius_km_negative_raises(self):
        with pytest.raises(ValidationError):
            InputData(latitude=0.0, longitude=0.0, radius_km=-5.0)


class TestAbstainBehaviour:
    def test_abstain_null_value_with_reason(self):
        gs = GroundedString(value=None, confidence=0.0, abstain_reason="no_recent_evidence", evidence=[])
        assert gs.value is None
        assert gs.abstain_reason == "no_recent_evidence"

    def test_abstain_empty_evidence_is_valid(self):
        gs = GroundedString(value=None, confidence=0.0, abstain_reason="not_found", evidence=[])
        assert gs.evidence == []

    def test_non_null_value_no_abstain(self):
        gs = GroundedString(value="Granite Quarry", confidence=0.85, abstain_reason=None, evidence=[
            Evidence(source_id="src_1", char_start=0, char_end=13, quote="Granite Quarry")
        ])
        assert gs.value is not None
        assert gs.abstain_reason is None


class TestSourceSchema:
    def test_valid_source(self):
        s = Source(
            source_id="src_1",
            url="https://example.com/quarry",
            fetched_at="2025-01-01T00:00:00Z",
            content_hash="sha256:abc123",
            trust_tier="official",
        )
        assert s.trust_tier == "official"

    def test_invalid_trust_tier(self):
        with pytest.raises(ValidationError):
            Source(
                source_id="src_1",
                url="https://example.com",
                fetched_at="2025-01-01T00:00:00Z",
                content_hash="sha256:abc",
                trust_tier="fake_tier",
            )

    def test_location_verification_confidence_range(self):
        with pytest.raises(ValidationError):
            LocationVerification(is_verified=True, confidence=1.5, method=LocationMethod.geocode)


class TestGroundedEnums:
    def test_valid_site_type(self):
        field = GroundedSiteType(value="Quarry", confidence=0.9, evidence=[])
        assert field.value == "Quarry"

    def test_invalid_site_type_raises(self):
        with pytest.raises(ValidationError):
            GroundedSiteType(value="Mine", confidence=0.9, evidence=[])

    def test_valid_operational_status(self):
        field = GroundedOperationalStatus(value="active", confidence=0.9, evidence=[])
        assert field.value == "active"

    def test_invalid_operational_status_raises(self):
        with pytest.raises(ValidationError):
            GroundedOperationalStatus(value="operational", confidence=0.9, evidence=[])
