# Tests for extraction logic - no LLM calls, no network, all offline
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

from pipeline.extractor import (
    _find_quote_positions,
    _build_evidence,
    _parse_grounded_field,
    extract_from_osm_tags,
)
from pipeline.scraper import _is_valid_url, _is_robots_allowed, _html_to_text


class TestQuotePositions:
    def test_exact_match(self):
        text = "Welcome to Carrière de Vignats quarry."
        start, end = _find_quote_positions(text, "Carrière de Vignats")
        assert start == 11
        assert end == 30

    def test_not_found_returns_zeros(self):
        start, end = _find_quote_positions("hello world", "missing phrase")
        assert start == 0
        assert end == 0

    def test_empty_quote(self):
        start, end = _find_quote_positions("some text", "")
        assert start == 0
        assert end == 0

    def test_case_insensitive_fallback(self):
        text = "This is a LIMESTONE quarry"
        start, end = _find_quote_positions(text, "limestone")
        assert start >= 0


class TestBuildEvidence:
    def test_valid_quote_found(self):
        content = "The Granite Works is an active quarry."
        ev = _build_evidence(["Granite Works"], "src_1", content)
        assert len(ev) == 1
        assert ev[0]["quote"] == "Granite Works"
        assert ev[0]["char_start"] == 4
        assert ev[0]["source_id"] == "src_1"

    def test_not_found_quote_skipped(self):
        content = "Some other text"
        ev = _build_evidence(["nonexistent phrase xyz"], "src_1", content)
        assert ev == []

    def test_empty_quotes_list(self):
        ev = _build_evidence([], "src_1", "content")
        assert ev == []


class TestParseGroundedField:
    def test_valid_field(self):
        raw = {"value": "Limestone Quarry", "confidence": 0.85, "abstain_reason": None, "evidence_quotes": ["Limestone Quarry"]}
        content = "Limestone Quarry is located in France."
        result = _parse_grounded_field(raw, "src_1", content)
        assert result["value"] == "Limestone Quarry"
        assert result["confidence"] == 0.85
        assert len(result["evidence"]) == 1

    def test_null_value_gets_abstain(self):
        raw = {"value": None, "confidence": 0.0, "abstain_reason": None, "evidence_quotes": []}
        result = _parse_grounded_field(raw, "src_1", "text")
        assert result["value"] is None
        assert result["abstain_reason"] == "not_found"

    def test_non_dict_input(self):
        result = _parse_grounded_field("bad input", "src_1", "text")
        assert result["value"] is None
        assert result["abstain_reason"] == "not_found"

    def test_confidence_clamped(self):
        raw = {"value": "test", "confidence": 1.5, "abstain_reason": None, "evidence_quotes": []}
        result = _parse_grounded_field(raw, "src_1", "test content here")
        assert result["confidence"] <= 1.0

    def test_confidence_floor_when_value_present(self):
        raw = {"value": "test", "confidence": 0.0, "abstain_reason": None, "evidence_quotes": []}
        result = _parse_grounded_field(raw, "src_1", "test content")
        assert result["confidence"] >= 0.05


class TestOsmExtraction:
    def test_name_extracted(self):
        tags = {"name": "Carrière du Nord", "landuse": "quarry", "material": "limestone"}
        result = extract_from_osm_tags(tags, "src_osm")
        assert result["official_name"]["value"] == "Carrière du Nord"
        assert result["official_name"]["confidence"] > 0

    def test_site_type_always_quarry(self):
        tags = {"landuse": "quarry"}
        result = extract_from_osm_tags(tags, "src_osm")
        assert result["site_type"]["value"] == "Quarry"

    def test_materials_split_by_semicolon(self):
        tags = {"name": "Test", "material": "limestone;granite;sand"}
        result = extract_from_osm_tags(tags, "src_osm")
        assert len(result["materials_produced"]) == 3
        vals = {m["value"] for m in result["materials_produced"]}
        assert "limestone" in vals
        assert "granite" in vals

    def test_end_date_means_inactive(self):
        tags = {"name": "Old Quarry", "end_date": "2005"}
        result = extract_from_osm_tags(tags, "src_osm")
        assert result["operational_status"]["value"] == "inactive"

    def test_disused_tag_means_inactive(self):
        tags = {"name": "Closed Quarry", "disused": "yes"}
        result = extract_from_osm_tags(tags, "src_osm")
        assert result["operational_status"]["value"] == "inactive"

    def test_no_status_info_abstains(self):
        tags = {"name": "Mystery Quarry"}
        result = extract_from_osm_tags(tags, "src_osm")
        assert result["operational_status"]["value"] is None
        assert result["operational_status"]["abstain_reason"] is not None


class TestScraper:
    def test_invalid_url_scheme(self):
        assert not _is_valid_url("ftp://example.com/page")

    def test_valid_https_url(self):
        assert _is_valid_url("https://example.com/quarry")

    def test_empty_url(self):
        assert not _is_valid_url("")

    def test_blocked_social_domain(self):
        assert not _is_valid_url("https://facebook.com/quarry-page")

    def test_html_to_text_strips_scripts(self):
        html = "<html><body><script>alert('x')</script><p>Quarry info here</p></body></html>"
        text = _html_to_text(html)
        assert "alert" not in text
        assert "Quarry info here" in text

    def test_robots_txt_checked_before_fetch(self):
        # robots check must run before any actual HTTP call
        with patch("pipeline.scraper._is_robots_allowed", return_value=False) as mock_robots:
            with patch("pipeline.scraper.requests.get") as mock_get:
                from pipeline.scraper import fetch_page
                result = fetch_page("https://example.com/quarry-page")
                mock_robots.assert_called_once()
                mock_get.assert_not_called()
                assert result is None

    def test_valid_fetch_returns_text(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.content = b"<html><body><p>A quarry site with limestone</p></body></html>"
        mock_resp.text = "<html><body><p>A quarry site with limestone</p></body></html>"
        mock_resp.iter_content.return_value = [b"<html><body><p>A quarry site with limestone</p></body></html>"]

        with patch("pipeline.scraper._is_valid_url", return_value=True):
            with patch("pipeline.scraper._is_robots_allowed", return_value=True):
                with patch("pipeline.scraper.requests.get", return_value=mock_resp):
                    with patch("pipeline.scraper.time.sleep"):
                        from pipeline.scraper import fetch_page
                        result = fetch_page("https://example.com/quarry")
                        assert result is not None
                        text, _ = result
                        assert "limestone" in text

    def test_redirect_target_validated_before_follow(self):
        redirect_resp = MagicMock()
        redirect_resp.status_code = 302
        redirect_resp.headers = {"Location": "https://facebook.com/quarry"}

        with patch("pipeline.scraper._is_valid_url", side_effect=lambda url: "facebook.com" not in url):
            with patch("pipeline.scraper._is_robots_allowed", return_value=True):
                with patch("pipeline.scraper.requests.get", return_value=redirect_resp) as mock_get:
                    with patch("pipeline.scraper.time.sleep"):
                        from pipeline.scraper import fetch_page
                        result = fetch_page("https://example.com/quarry")
                        assert result is None
                        assert mock_get.call_count == 1

    def test_redirect_target_robots_checked_before_follow(self):
        redirect_resp = MagicMock()
        redirect_resp.status_code = 302
        redirect_resp.headers = {"Location": "/next"}

        with patch("pipeline.scraper._is_valid_url", return_value=True):
            with patch("pipeline.scraper._is_robots_allowed", side_effect=[True, False]) as mock_robots:
                with patch("pipeline.scraper.requests.get", return_value=redirect_resp) as mock_get:
                    with patch("pipeline.scraper.time.sleep"):
                        from pipeline.scraper import fetch_page
                        result = fetch_page("https://example.com/quarry")
                        assert result is None
                        assert mock_get.call_count == 1
                        assert mock_robots.call_count == 2


class TestAPIEndpoints:
    # smoke tests for all 5 endpoints using mocked database

    def _mock_db(self):
        mock = MagicMock()
        mock.query.return_value.filter_by.return_value.first.return_value = None
        mock.query.return_value.filter.return_value = mock.query.return_value
        mock.query.return_value.count.return_value = 0
        mock.query.return_value.offset.return_value.limit.return_value.all.return_value = []
        return mock

    def test_health_endpoint_returns_200(self, app_client):
        mock_db = self._mock_db()
        with patch("main.get_db", return_value=mock_db):
            with patch("main.sa_text"):
                with patch("redis.from_url") as mock_redis:
                    mock_redis.return_value.ping.return_value = True
                    mock_redis.return_value.llen.return_value = 0
                    resp = app_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "queue_depth" in data

    def test_get_job_not_found_returns_404(self, app_client):
        mock_db = self._mock_db()
        with patch("main.get_db", return_value=mock_db):
            resp = app_client.get("/api/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_get_job_found_returns_200(self, app_client):
        from datetime import datetime
        mock_db = self._mock_db()
        job_mock = MagicMock()
        job_mock.id = "test-job-id"
        job_mock.status = "completed"
        job_mock.progress = 100
        job_mock.result_count = 3
        job_mock.error = None
        job_mock.created_at = datetime.utcnow()
        job_mock.updated_at = datetime.utcnow()
        mock_db.query.return_value.filter_by.return_value.first.return_value = job_mock
        with patch("main.get_db", return_value=mock_db):
            resp = app_client.get("/api/jobs/test-job-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "completed"

    def test_sites_list_returns_200(self, app_client):
        mock_db = self._mock_db()
        with patch("main.get_db", return_value=mock_db):
            resp = app_client.get("/api/sites")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_site_not_found_returns_404(self, app_client):
        mock_db = self._mock_db()
        with patch("main.get_db", return_value=mock_db):
            resp = app_client.get("/api/sites/nonexistent-site")
        assert resp.status_code == 404
