# Focused regression tests for discovery enrichment and stable reconciliation output.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch

from pipeline.discovery import QuarryCandidate, _enrich_candidates_with_search, _looks_like_same_site
from pipeline.extractor import extract_from_osm_tags
from pipeline.reconciler import reconcile


class TestDiscoveryEnrichment:
    def test_name_matching_is_accent_insensitive(self):
        assert _looks_like_same_site(
            "Carriere de Vignats",
            "Carriere de Vignats - site officiel",
            "Extraction de calcaire en Normandie",
        )

    def test_search_results_only_attach_to_matching_candidate(self):
        candidates = [
            QuarryCandidate(
                osm_id="node/1",
                name="Carriere de Vignats",
                latitude=48.0,
                longitude=2.0,
                candidate_urls=["https://www.openstreetmap.org/node/1"],
            ),
            QuarryCandidate(
                osm_id="node/2",
                name="Carrieres du Boulonnais",
                latitude=49.0,
                longitude=1.0,
                candidate_urls=["https://www.openstreetmap.org/node/2"],
            ),
        ]

        def fake_serper(query, num=5):
            if "Vignats" in query:
                return [{"title": "Carriere de Vignats", "snippet": "Site officiel", "link": "https://vignats.example.com"}]
            return [{"title": "Carrieres du Boulonnais", "snippet": "Calcaire et craie", "link": "https://boulonnais.example.com"}]

        with patch("pipeline.discovery.SERPER_API_KEY", "test-key"):
            with patch("pipeline.discovery._serper_search", side_effect=fake_serper):
                _enrich_candidates_with_search(candidates, "Caen")

        assert "https://vignats.example.com" in candidates[0].candidate_urls
        assert "https://boulonnais.example.com" not in candidates[0].candidate_urls
        assert "https://boulonnais.example.com" in candidates[1].candidate_urls


class TestReconciliationIdentity:
    def test_site_id_and_prompt_hash_are_stable_for_same_candidate(self):
        tags = {"name": "Carriere de Vignats", "landuse": "quarry", "material": "limestone"}
        candidate = QuarryCandidate(
            osm_id="node/123",
            name="Carriere de Vignats",
            latitude=48.8566,
            longitude=2.3522,
            osm_tags=tags,
            candidate_urls=[],
        )
        osm_extraction = extract_from_osm_tags(tags, "src_osm_0")
        common_kwargs = {
            "candidate": candidate,
            "scraped_extractions": [],
            "osm_extraction": osm_extraction,
            "osm_source_id": "src_osm_0",
            "model_calls": [],
            "input_data": {"latitude": 48.8566, "longitude": 2.3522, "radius_km": 50},
            "latency_ms": 100,
            "fetched_at": "2026-05-04T00:00:00Z",
            "source_content_hashes": {"src_osm_0": "sha256:osm"},
            "expected_city": None,
        }

        record_one = reconcile(**common_kwargs)
        record_two = reconcile(**common_kwargs)

        assert record_one.site_id == record_two.site_id
        assert record_one.run_metadata.prompt_hash == record_two.run_metadata.prompt_hash
