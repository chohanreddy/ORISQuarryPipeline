# LLM extraction using Google Gemini Flash (free tier, 15 RPM / 1M tokens per day)
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Gemini 2.0 Flash pricing (prompts under 128k tokens)
# $0.10 per million input tokens, $0.40 per million output tokens
PRICE_INPUT_PER_M  = 0.10
PRICE_OUTPUT_PER_M = 0.40

SCRAPER_VERSION = "1.0.0"

EXTRACTION_SYSTEM = """You are an expert data extraction agent for ORIS, a global database of construction material sites.
Your task: extract structured information about quarry sites from web page content.

Rules:
- Only extract information explicitly stated in the provided text. Never invent or infer.
- If information is absent or uncertain, set value to null and provide an abstain_reason.
- abstain_reason must be one of: "not_found", "ambiguous", "stale_source", "no_recent_evidence".
- evidence_quotes must be verbatim substrings copied exactly from the source text.
- operational_status: use "active", "inactive", "unknown", or null only.
- site_type: use "Quarry" or null only.
- All confidence values must be between 0.0 and 1.0.
"""

EXTRACTION_PROMPT = """Source ID: {source_id}
Source URL: {url}
Page last-modified: {last_modified}

Web page content:
---
{content}
---

Extract quarry information. Return ONLY a JSON object with this exact structure:

{{
  "official_name": {{
    "value": "string or null",
    "confidence": 0.0,
    "abstain_reason": "string or null",
    "evidence_quotes": ["verbatim quote from text"]
  }},
  "site_type": {{
    "value": "Quarry" or null,
    "confidence": 0.0,
    "abstain_reason": "string or null",
    "evidence_quotes": ["verbatim quote from text"]
  }},
  "description": {{
    "value": "string or null",
    "confidence": 0.0,
    "abstain_reason": "string or null",
    "evidence_quotes": ["verbatim quote from text"]
  }},
  "materials_produced": [
    {{
      "value": "material name",
      "confidence": 0.0,
      "abstain_reason": null,
      "evidence_quotes": ["verbatim quote"]
    }}
  ],
  "certifications": [],
  "operational_status": {{
    "value": "active" or "inactive" or "unknown" or null,
    "confidence": 0.0,
    "abstain_reason": "string or null",
    "evidence_quotes": ["verbatim quote from text"]
  }},
  "location_hint": "city or region name mentioned in the text, or null"
}}

For operational_status, apply these recency rules:
- If page last-modified is older than 3 years and the content contains no dates from the past 2 years, use abstain_reason "stale_source" with value null.
- If you find dated evidence of recent activity (job listings, news, events within 2 years), set "active" with higher confidence.
- Only set "inactive" if you see explicit closure evidence (e.g. "fermé", "closed", "end_date").
- The most recent year mentioned in the content is a useful recency signal."""


def _find_quote_positions(text: str, quote: str) -> Tuple[int, int]:
    if not quote or not text:
        return 0, 0
    idx = text.find(quote)
    if idx == -1:
        idx = text.lower().find(quote.lower())
    if idx == -1:
        return 0, 0
    return idx, idx + len(quote)


def _build_evidence(quotes: List[str], source_id: str, content: str) -> List[Dict]:
    evidence = []
    for q in quotes:
        if not q or not q.strip():
            continue
        start, end = _find_quote_positions(content, q)
        if start == 0 and end == 0 and q.strip() not in content:
            continue
        evidence.append({
            "source_id": source_id,
            "char_start": start,
            "char_end": end,
            "quote": q[:500],
        })
    return evidence


def _parse_grounded_field(raw: Any, source_id: str, content: str) -> Dict:
    if not isinstance(raw, dict):
        return {"value": None, "confidence": 0.0, "abstain_reason": "not_found", "evidence": []}
    value = raw.get("value")
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    abstain = raw.get("abstain_reason")
    quotes = raw.get("evidence_quotes", [])
    if not isinstance(quotes, list):
        quotes = []
    evidence = _build_evidence(quotes, source_id, content)
    if value is None and not abstain:
        abstain = "not_found"
    if value is not None and confidence < 0.05:
        confidence = 0.05
    return {"value": value, "confidence": confidence, "abstain_reason": abstain, "evidence": evidence}


def _sanitize_enum_field(gd: Dict, allowed_values: set[str]) -> Dict:
    value = gd.get("value")
    if value is None:
        return gd
    if value in allowed_values:
        return gd

    return {
        "value": None,
        "confidence": 0.0,
        "abstain_reason": "ambiguous",
        "evidence": gd.get("evidence", []),
    }


def _calculate_cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in / 1_000_000) * PRICE_INPUT_PER_M + (tokens_out / 1_000_000) * PRICE_OUTPUT_PER_M


def _empty_extraction(model: str) -> Tuple[Dict, Dict]:
    empty = {
        "official_name":     {"value": None, "confidence": 0.0, "abstain_reason": "not_found", "evidence": []},
        "site_type":         {"value": None, "confidence": 0.0, "abstain_reason": "not_found", "evidence": []},
        "description":       {"value": None, "confidence": 0.0, "abstain_reason": "not_found", "evidence": []},
        "materials_produced": [],
        "certifications":    [],
        "operational_status":{"value": None, "confidence": 0.0, "abstain_reason": "not_found", "evidence": []},
        "location_hint": None,
    }
    call = {"model": model, "purpose": "extraction", "tokens_in": 0, "tokens_out": 0, "usd_cost": 0.0}
    return empty, call


def extract_from_source(
    content: str,
    source_id: str,
    url: str,
    last_modified: Optional[str] = None,
    max_content_chars: int = 12000,
) -> Tuple[Dict, Dict]:
    # runs Gemini on scraped text, returns (extraction_dict, model_call_dict)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set - skipping LLM extraction for %s", source_id)
        return _empty_extraction(GEMINI_MODEL)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=EXTRACTION_SYSTEM,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1500,
        ),
    )

    truncated = content[:max_content_chars]
    prompt = EXTRACTION_PROMPT.format(
        source_id=source_id,
        url=url,
        content=truncated,
        last_modified=last_modified or "unknown",
    )

    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        return _empty_extraction(GEMINI_MODEL)

    # usage_metadata gives us token counts for cost tracking
    usage = getattr(response, "usage_metadata", None)
    tokens_in  = getattr(usage, "prompt_token_count", 0) or 0
    tokens_out = getattr(usage, "candidates_token_count", 0) or 0
    cost = _calculate_cost(tokens_in, tokens_out)

    model_call = {
        "model": GEMINI_MODEL,
        "purpose": f"extraction:{source_id}",
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "usd_cost": cost,
    }

    raw_text = response.text.strip()

    # gemini sometimes wraps its output in ```json ... ``` fences, strip those out
    raw_text = re.sub(r"^```json\s*", "", raw_text)
    raw_text = re.sub(r"^```\s*",     "", raw_text)
    raw_text = re.sub(r"\s*```$",     "", raw_text)

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini JSON for %s: %s", source_id, raw_text[:200])
        raw = {}

    extraction = {
        "official_name": _parse_grounded_field(raw.get("official_name"), source_id, truncated),
        "site_type":     _sanitize_enum_field(
            _parse_grounded_field(raw.get("site_type"), source_id, truncated),
            {"Quarry"},
        ),
        "description":   _parse_grounded_field(raw.get("description"),   source_id, truncated),
        "materials_produced": [
            _parse_grounded_field(m, source_id, truncated)
            for m in (raw.get("materials_produced") or [])
            if isinstance(m, dict)
        ],
        "certifications": [
            _parse_grounded_field(c, source_id, truncated)
            for c in (raw.get("certifications") or [])
            if isinstance(c, dict)
        ],
        "operational_status": _sanitize_enum_field(
            _parse_grounded_field(raw.get("operational_status"), source_id, truncated),
            {"active", "inactive", "unknown"},
        ),
        "location_hint": raw.get("location_hint"),
    }

    return extraction, model_call


def extract_from_osm_tags(tags: dict, source_id: str) -> Dict:
    # free baseline extraction straight from OSM tags, no LLM call needed
    name        = tags.get("name") or tags.get("official_name")
    material    = tags.get("material") or tags.get("substance")
    operator    = tags.get("operator")
    description = tags.get("description")
    start_date  = tags.get("start_date")
    end_date    = tags.get("end_date")

    disused    = tags.get("disused") or tags.get("abandoned") or "disused" in tags.get("historic", "")
    active_tag = tags.get("operational_status") or tags.get("status")

    if end_date:
        op_status, op_conf, op_reason = "inactive", 0.7, None
    elif disused or active_tag in ("closed", "abandoned", "disused"):
        op_status, op_conf, op_reason = "inactive", 0.8, None
    elif active_tag in ("active", "operational"):
        op_status, op_conf, op_reason = "active", 0.8, None
    elif start_date and not end_date:
        op_status, op_conf, op_reason = "unknown", 0.3, "no_recent_evidence"
    else:
        op_status, op_conf, op_reason = None, 0.0, "not_found"

    content = "\n".join(f"{k}: {v}" for k, v in tags.items())

    def _build(value, conf, reason, _key):
        if value:
            start, end = _find_quote_positions(content, str(value))
            ev = [{"source_id": source_id, "char_start": start, "char_end": end, "quote": str(value)}]
        else:
            ev = []
        return {"value": value, "confidence": conf, "abstain_reason": reason, "evidence": ev}

    materials = []
    if material:
        for mat in re.split(r"[;,/]", material):
            mat = mat.strip()
            if mat:
                start, end = _find_quote_positions(content, mat)
                materials.append({
                    "value": mat, "confidence": 0.85, "abstain_reason": None,
                    "evidence": [{"source_id": source_id, "char_start": start, "char_end": end, "quote": mat}],
                })

    return {
        "official_name": _build(name, 0.90 if name else 0.0, None if name else "not_found", "name"),
        "site_type":     _build("Quarry", 0.90, None, "landuse"),
        "description":   _build(
            description or (f"Operated by {operator}" if operator else None),
            0.7 if (description or operator) else 0.0,
            None if (description or operator) else "not_found", "description"
        ),
        "materials_produced": materials,
        "certifications":     [],
        "operational_status": _build(op_status, op_conf, op_reason, "status"),
        "location_hint":      None,
    }
