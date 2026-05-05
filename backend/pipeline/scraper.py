# Web scraper - checks robots.txt first, respects rate limits, strips HTML to plain text
from __future__ import annotations
import hashlib
import logging
import random
import time
import urllib.parse
import urllib.robotparser
from email.utils import parsedate_to_datetime
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = "OrisQuarryPipeline/1.0 (contact: pipeline@oris.example)"
REQUEST_TIMEOUT = 20
MAX_CONTENT_BYTES = 500_000
MAX_TEXT_CHARS = 15_000

ALLOWED_SCHEMES = {"http", "https"}

# social media sites dont have useful quarry content, just block them upfront
BLOCKLIST = {
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
    "youtube.com", "tiktok.com",
}

_robots_cache: dict = {}


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ALLOWED_SCHEMES:
            return False
        if not parsed.netloc:
            return False
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        for blocked in BLOCKLIST:
            if host == blocked or host.endswith("." + blocked):
                return False
        return True
    except Exception:
        return False


def _get_robots(base_url: str) -> urllib.robotparser.RobotFileParser:
    if base_url in _robots_cache:
        return _robots_cache[base_url]
    rp = urllib.robotparser.RobotFileParser()
    robots_url = base_url.rstrip("/") + "/robots.txt"
    try:
        resp = requests.get(
            robots_url,
            headers={"User-Agent": USER_AGENT},
            timeout=10,
            allow_redirects=True,
        )
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            # no robots.txt found, assume everything is allowed
            rp.parse([])
    except Exception:
        rp.parse([])
    _robots_cache[base_url] = rp
    return rp


def _is_robots_allowed(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = _get_robots(base)
    return rp.can_fetch(USER_AGENT, url)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return text[:MAX_TEXT_CHARS]


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _parse_last_modified(value: str) -> Optional[str]:
    try:
        return parsedate_to_datetime(value).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _read_limited_body(resp: requests.Response) -> bytes:
    chunks = []
    total = 0

    try:
        for chunk in resp.iter_content(chunk_size=16_384):
            if not chunk:
                continue

            remaining = MAX_CONTENT_BYTES - total
            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk[:remaining]

            chunks.append(chunk)
            total += len(chunk)

            if total >= MAX_CONTENT_BYTES:
                break
    finally:
        resp.close()

    return b"".join(chunks)


def fetch_page(url: str) -> Optional[Tuple[str, str, Optional[str]]]:
    # returns (clean_text, content_hash, last_modified_iso) or None if we cant/shouldnt fetch it
    if not _is_valid_url(url):
        logger.debug("Invalid or blocked URL: %s", url)
        return None

    if not _is_robots_allowed(url):
        logger.info("robots.txt disallows: %s", url)
        return None

    time.sleep(random.uniform(0.5, 2.0))

    current_url = url
    redirects_followed = 0

    for attempt in range(3):
        try:
            resp = requests.get(
                current_url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en,fr;q=0.9",
                },
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
                stream=True,
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Fetch failed (%s): %s", current_url, exc)
            return None

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            retry_after = min(retry_after, 120)
            logger.info("Rate limited on %s, sleeping %ds", current_url, retry_after)
            resp.close()
            time.sleep(retry_after + random.uniform(1, 5))
            continue

        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location", "")
            next_url = urllib.parse.urljoin(current_url, location)
            resp.close()

            if not _is_valid_url(next_url):
                logger.warning("Redirect blocked: %s -> %s", current_url, next_url)
                return None

            if not _is_robots_allowed(next_url):
                logger.info("robots.txt disallows redirect target: %s", next_url)
                return None

            redirects_followed += 1
            if redirects_followed > 5:
                logger.warning("Too many redirects while fetching %s", url)
                return None

            current_url = next_url
            continue

        if resp.status_code != 200:
            logger.debug("Non-200 status %d for %s", resp.status_code, current_url)
            resp.close()
            return None

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            logger.debug("Non-HTML content type %s for %s", content_type, current_url)
            resp.close()
            return None

        raw = _read_limited_body(resp).decode("utf-8", errors="replace")
        text = _html_to_text(raw)
        if not text.strip():
            return None

        last_modified = _parse_last_modified(resp.headers.get("Last-Modified", ""))
        return text, content_hash(text), last_modified

    return None


def fetch_osm_page_text(osm_tags: dict) -> str:
    # just dump the OSM tags as key: value pairs, no HTTP call needed
    parts = []
    for key, val in osm_tags.items():
        parts.append(f"{key}: {val}")
    return "\n".join(parts)
