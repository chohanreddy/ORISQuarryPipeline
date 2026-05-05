# Quarry discovery - pulls candidates from OSM Overpass and optionally Serper web search
from __future__ import annotations
import os
import math
import logging
import time
import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)

# try multiple endpoints in case one is down - overpass is sometimes flaky
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
SERPER_URL = "https://google.serper.dev/search"
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
USER_AGENT = "OrisQuarryPipeline/1.0 (contact: pipeline@oris.example)"
SERPER_RESULTS_PER_QUERY = 5
SERPER_MAX_ENRICHED_CANDIDATES = 10


@dataclass
class QuarryCandidate:
    osm_id: Optional[str]
    name: str
    latitude: float
    longitude: float
    osm_tags: dict = field(default_factory=dict)
    candidate_urls: List[str] = field(default_factory=list)
    trust_tier: str = "unknown"


def _km_to_deg_lat(km: float) -> float:
    return km / 111.0


def _bounding_box(lat: float, lon: float, radius_km: float):
    lat_delta = _km_to_deg_lat(radius_km)
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
    return (lat - lat_delta, lon - lon_delta, lat + lat_delta, lon + lon_delta)


def _overpass_query(lat: float, lon: float, radius_km: float) -> List[QuarryCandidate]:
    radius_m = int(radius_km * 1000)
    query = f"""
[out:json][timeout:60];
(
  node["landuse"="quarry"](around:{radius_m},{lat},{lon});
  way["landuse"="quarry"](around:{radius_m},{lat},{lon});
  relation["landuse"="quarry"](around:{radius_m},{lat},{lon});
  node["man_made"="quarry"](around:{radius_m},{lat},{lon});
  way["man_made"="quarry"](around:{radius_m},{lat},{lon});
  node["quarry"](around:{radius_m},{lat},{lon});
  way["quarry"](around:{radius_m},{lat},{lon});
);
out center tags meta;
"""
    data = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            logger.info("Trying Overpass endpoint: %s", endpoint)
            resp = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            logger.warning("Overpass endpoint %s failed: %s - trying next", endpoint, exc)
            time.sleep(random.uniform(1, 3))

    if data is None:
        logger.error("All Overpass endpoints failed")
        return []

    candidates: List[QuarryCandidate] = []
    seen_ids: set = set()

    for element in data.get("elements", []):
        osm_id = f"{element['type']}/{element['id']}"
        if osm_id in seen_ids:
            continue
        seen_ids.add(osm_id)

        tags = element.get("tags", {})
        name = (
            tags.get("name")
            or tags.get("official_name")
            or tags.get("operator")
            or f"Unnamed quarry ({osm_id})"
        )

        if element["type"] == "node":
            c_lat, c_lon = element.get("lat", lat), element.get("lon", lon)
        else:
            center = element.get("center", {})
            c_lat = center.get("lat", lat)
            c_lon = center.get("lon", lon)

        urls: List[str] = []
        if tags.get("website"):
            urls.append(tags["website"].strip())
        if tags.get("url"):
            urls.append(tags["url"].strip())
        if tags.get("contact:website"):
            urls.append(tags["contact:website"].strip())

        osm_page = f"https://www.openstreetmap.org/{element['type']}/{element['id']}"
        urls.append(osm_page)

        candidates.append(
            QuarryCandidate(
                osm_id=osm_id,
                name=name,
                latitude=c_lat,
                longitude=c_lon,
                osm_tags=tags,
                candidate_urls=_dedupe_urls(urls),
                trust_tier="official",
            )
        )

    logger.info("OSM returned %d quarry candidates", len(candidates))
    return candidates


def _normalize_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def _name_tokens(text: str) -> set[str]:
    stopwords = {
        "career", "carriere", "carrieres", "quarry", "site", "mine",
        "de", "du", "des", "la", "le", "les", "the", "and",
    }
    return {
        token
        for token in _normalize_text(text).split()
        if len(token) > 2 and token not in stopwords
    }


def _looks_like_same_site(candidate_name: str, title: str, snippet: str) -> bool:
    normalized_candidate = _normalize_text(candidate_name)
    normalized_result = _normalize_text(f"{title} {snippet}")
    if not normalized_candidate or not normalized_result:
        return False

    if normalized_candidate in normalized_result:
        return True

    candidate_tokens = _name_tokens(candidate_name)
    result_tokens = _name_tokens(f"{title} {snippet}")
    if not candidate_tokens or not result_tokens:
        return False

    overlap = len(candidate_tokens & result_tokens)
    minimum_overlap = 1 if len(candidate_tokens) == 1 else 2
    return overlap >= minimum_overlap


def _serper_search(query: str, num: int = SERPER_RESULTS_PER_QUERY) -> List[dict]:
    if not SERPER_API_KEY:
        return []

    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "User-Agent": USER_AGENT},
            json={"q": query, "num": num},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("organic", [])
    except Exception as exc:
        logger.warning("Serper search failed: %s", exc)
        return []

    logger.info("Serper returned %d results for query %r", len(results), query)
    return results


def _reverse_geocode_city(lat: float, lon: float) -> Optional[str]:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        addr = resp.json().get("address", {})
        return (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("county")
        )
    except Exception:
        return None


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _build_serper_query(candidate_name: str, city: Optional[str]) -> str:
    if city:
        return f'"{candidate_name}" quarry {city}'
    return f'"{candidate_name}" quarry'


def _serper_area_discover(lat: float, lon: float, city: Optional[str]) -> List[QuarryCandidate]:
    """Search for quarries in the area via web search — catches sites absent from OSM."""
    if not SERPER_API_KEY:
        return []

    queries = []
    if city:
        queries.append(f'quarry {city}')
        queries.append(f'carrière {city}')
    else:
        queries.append(f'quarry {lat:.3f} {lon:.3f}')

    candidates: List[QuarryCandidate] = []
    seen_urls: set = set()

    for query in queries[:2]:
        for result in _serper_search(query):
            url = result.get('link', '').strip()
            title = result.get('title', '')
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            name = title.split(' - ')[0].split(' | ')[0].strip() or title
            candidates.append(QuarryCandidate(
                osm_id=None,
                name=name,
                latitude=lat,
                longitude=lon,
                osm_tags={},
                candidate_urls=[url],
                trust_tier='directory',
            ))

    logger.info("Serper area search found %d candidates", len(candidates))
    return candidates


def _merge_candidates(
    osm: List[QuarryCandidate], web: List[QuarryCandidate]
) -> List[QuarryCandidate]:
    """Append web-discovered candidates that don't match any existing OSM entry."""
    merged = list(osm)
    for wc in web:
        if not any(_looks_like_same_site(oc.name, wc.name, '') for oc in osm):
            merged.append(wc)
    return merged


def _enrich_candidates_with_search(candidates: List[QuarryCandidate], city: Optional[str]) -> None:
    if not SERPER_API_KEY:
        return

    for candidate in candidates[:SERPER_MAX_ENRICHED_CANDIDATES]:
        if not candidate.name or candidate.name.startswith("Unnamed quarry"):
            continue

        query = _build_serper_query(candidate.name, city)
        results = _serper_search(query)
        matched_urls: List[str] = []

        for result in results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("link", "").strip()
            if not url:
                continue
            if _looks_like_same_site(candidate.name, title, snippet):
                matched_urls.append(url)

        if matched_urls:
            candidate.candidate_urls = _dedupe_urls(candidate.candidate_urls + matched_urls)


def discover_quarries(lat: float, lon: float, radius_km: float) -> List[QuarryCandidate]:
    time.sleep(random.uniform(0.3, 0.8))  # small jitter before hitting external APIs
    osm_candidates = _overpass_query(lat, lon, radius_km)
    city = _reverse_geocode_city(lat, lon) if (SERPER_API_KEY or osm_candidates) else None
    web_candidates = _serper_area_discover(lat, lon, city)
    all_candidates = _merge_candidates(osm_candidates, web_candidates)
    _enrich_candidates_with_search(all_candidates, city)
    logger.info(
        "Total candidates: %d (OSM: %d, web-only: %d)",
        len(all_candidates), len(osm_candidates), len(web_candidates),
    )
    return all_candidates
