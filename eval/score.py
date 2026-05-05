#!/usr/bin/env python3
# Scoring script - runs the pipeline against ground truth and reports field-level accuracy.
#
# Usage:
#     python eval/score.py [--api-url http://localhost:8000] [--wait-sec 180]
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import jsonschema

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.json"

OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["site_id", "schema_version", "input", "extraction", "provenance", "metrics", "run_metadata"],
    "properties": {
        "site_id": {"type": "string"},
        "schema_version": {"const": "2.0.0"},
        "input": {
            "type": "object",
            "required": ["latitude", "longitude", "radius_km"],
            "properties": {
                "latitude":  {"type": "number", "minimum": -90,  "maximum": 90},
                "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                "radius_km": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "extraction": {
            "type": "object",
            "properties": {
                "official_name": {"$ref": "#/$defs/groundedString"},
                "site_type": {"$ref": "#/$defs/groundedSiteType"},
                "description": {"$ref": "#/$defs/groundedString"},
                "materials_produced": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/groundedString"},
                },
                "certifications": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/groundedString"},
                },
                "operational_status": {"$ref": "#/$defs/groundedOperationalStatus"},
                "location_verification": {
                    "type": "object",
                    "required": ["is_verified", "confidence", "method"],
                    "properties": {
                        "is_verified": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "extracted_city": {"type": ["string", "null"]},
                        "method": {"enum": ["string_match", "geocode", "llm_inference", "none"]},
                    },
                },
            },
        },
        "provenance": {
            "type": "object",
            "required": ["sources", "reconciliations"],
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["source_id", "url", "fetched_at", "content_hash"],
                        "properties": {
                            "source_id": {"type": "string"},
                            "url": {"type": "string", "format": "uri"},
                            "fetched_at": {"type": "string", "format": "date-time"},
                            "content_hash": {"type": "string"},
                            "trust_tier": {"enum": ["official", "directory", "news", "unknown"]},
                        },
                    },
                },
                "reconciliations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["field", "candidates", "winner_source_id", "reason"],
                        "properties": {
                            "field": {"type": "string"},
                            "candidates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["source_id", "score"],
                                    "properties": {
                                        "value": {},
                                        "source_id": {"type": "string"},
                                        "score": {"type": "number"},
                                    },
                                },
                            },
                            "winner_source_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
        },
        "metrics": {
            "type": "object",
            "required": ["llm_tokens_in", "llm_tokens_out", "usd_cost", "latency_ms", "model_calls"],
            "properties": {
                "llm_tokens_in": {"type": "integer"},
                "llm_tokens_out": {"type": "integer"},
                "usd_cost": {"type": "number"},
                "latency_ms": {"type": "integer"},
                "model_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["model", "purpose", "tokens_in", "tokens_out", "usd_cost"],
                        "properties": {
                            "model": {"type": "string"},
                            "purpose": {"type": "string"},
                            "tokens_in": {"type": "integer"},
                            "tokens_out": {"type": "integer"},
                            "usd_cost": {"type": "number"},
                        },
                    },
                },
            },
        },
        "run_metadata": {
            "type": "object",
            "required": ["run_id", "prompt_hash", "scraper_version", "created_at"],
            "properties": {
                "run_id": {"type": "string"},
                "prompt_hash": {"type": "string"},
                "scraper_version": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
            },
        },
    },
    "$defs": {
        "evidence": {
            "type": "object",
            "required": ["source_id", "char_start", "char_end", "quote"],
            "properties": {
                "source_id": {"type": "string"},
                "char_start": {"type": "integer", "minimum": 0},
                "char_end": {"type": "integer", "minimum": 0},
                "quote": {"type": "string"},
            },
        },
        "groundedString": {
            "type": "object",
            "required": ["value", "confidence", "evidence"],
            "properties": {
                "value": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "abstain_reason": {"type": ["string", "null"]},
                "evidence": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/evidence"},
                },
            },
        },
        "groundedSiteType": {
            "allOf": [{"$ref": "#/$defs/groundedString"}],
            "properties": {
                "value": {"enum": ["Quarry", None]},
            },
        },
        "groundedOperationalStatus": {
            "allOf": [{"$ref": "#/$defs/groundedString"}],
            "properties": {
                "value": {"enum": ["active", "inactive", "unknown", None]},
            },
        },
    },
}


def submit_job(api_url: str, input_data: dict) -> Optional[str]:
    try:
        r = requests.post(f"{api_url}/api/jobs", json=input_data, timeout=30)
        r.raise_for_status()
        return r.json()["job_id"]
    except Exception as e:
        print(f"  [ERROR] Job submission failed: {e}")
        return None


def poll_job(api_url: str, job_id: str, wait_sec: int) -> Optional[dict]:
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        try:
            r = requests.get(f"{api_url}/api/jobs/{job_id}", timeout=10)
            r.raise_for_status()
            job = r.json()
            status = job["status"]
            progress = job.get("progress", 0)
            print(f"  ... {status} ({progress}%)", end="\r", flush=True)
            if status in ("completed", "failed"):
                print()
                return job
        except Exception:
            pass
        time.sleep(3)
    print()
    return None


def fetch_sites_for_job(api_url: str, job_id: str) -> List[dict]:
    results = []
    page = 1
    page_size = 50
    try:
        while True:
            r = requests.get(f"{api_url}/api/sites", params={"page": page, "page_size": page_size}, timeout=10)
            r.raise_for_status()
            body = r.json()
            items = body["items"]
            results.extend(s for s in items if s.get("job_id") == job_id)
            if len(items) < page_size:
                break
            page += 1
    except Exception as e:
        print(f"  [ERROR] Fetching sites: {e}")
    return results


def fetch_full_site(api_url: str, site_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{api_url}/api/sites/{site_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def validate_schema(record: dict) -> List[str]:
    errors = []
    try:
        jsonschema.validate(record, OUTPUT_SCHEMA)
    except jsonschema.ValidationError as e:
        errors.append(str(e.message))
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e}")
    extraction = record.get("extraction", {})
    for field in ["official_name", "site_type", "description", "operational_status"]:
        gf = extraction.get(field)
        if gf is not None:
            if not isinstance(gf.get("confidence"), (int, float)):
                errors.append(f"{field}.confidence must be a number")
            elif not (0.0 <= gf["confidence"] <= 1.0):
                errors.append(f"{field}.confidence={gf['confidence']} out of range [0,1]")
            if gf.get("value") is None and not gf.get("abstain_reason"):
                errors.append(f"{field}: value is null but abstain_reason is missing")
    site_type = extraction.get("site_type") or {}
    if site_type.get("value") not in (None, "Quarry"):
        errors.append("site_type.value must be 'Quarry' or null")
    op_status = extraction.get("operational_status") or {}
    if op_status.get("value") not in (None, "active", "inactive", "unknown"):
        errors.append("operational_status.value must be active/inactive/unknown/null")
    return errors


def score_site(site_record: dict, expected: dict) -> Dict[str, Any]:
    extraction = site_record.get("extraction", {})

    def _val(field: str) -> Optional[str]:
        gf = extraction.get(field)
        if gf is None:
            return None
        return gf.get("value")

    name_val = (_val("official_name") or "").lower()
    status_val = _val("operational_status")
    materials = [(m.get("value") or "").lower() for m in (extraction.get("materials_produced") or [])]

    expected_name = expected.get("official_name", "")
    expected_status = expected.get("operational_status")
    expected_mats = [m.lower() for m in (expected.get("materials_produced") or [])]

    name_match = bool(expected_name and expected_name.lower() in name_val)
    status_match = (expected_status is None) or (status_val == expected_status)
    mats_match = all(any(em in m for m in materials) for em in expected_mats) if expected_mats else True

    score = sum([name_match, status_match, mats_match]) / 3.0

    return {
        "name_match": name_match,
        "name_extracted": _val("official_name"),
        "status_match": status_match,
        "status_extracted": status_val,
        "materials_match": mats_match,
        "materials_extracted": materials,
        "field_score": round(score, 3),
    }


def print_separator():
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="ORIS quarry pipeline evaluator")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Backend API URL")
    parser.add_argument("--wait-sec", type=int, default=300, help="Max seconds to wait per job")
    args = parser.parse_args()

    ground_truth = json.loads(GROUND_TRUTH_FILE.read_text())

    total_field_scores = []
    schema_errors_total = []

    print("\n" + "=" * 70)
    print("  ORIS Quarry Pipeline - Ground Truth Evaluation")
    print("=" * 70)

    for gt in ground_truth:
        print_separator()
        print(f"\nCase: {gt['label']}")
        print(f"  Input: lat={gt['input']['latitude']}, lon={gt['input']['longitude']}, r={gt['input']['radius_km']}km")

        job_id = submit_job(args.api_url, gt["input"])
        if not job_id:
            print("  SKIPPED (submission failed)")
            continue

        print(f"  Job ID: {job_id}")
        job = poll_job(args.api_url, job_id, args.wait_sec)

        if not job:
            print("  TIMEOUT - job did not complete in time")
            continue
        if job["status"] == "failed":
            print(f"  FAILED - {job.get('error')}")
            continue

        sites = fetch_sites_for_job(args.api_url, job_id)
        print(f"  Sites extracted: {len(sites)}")

        if not sites:
            print("  No sites - scoring 0 for this case")
            total_field_scores.append(0.0)
            continue

        best_score = None
        for site in sites:
            full = fetch_full_site(args.api_url, site["id"])
            if not full:
                continue

            schema_errors = validate_schema(full)
            if schema_errors:
                print(f"  [SCHEMA ERROR] site {site['id']}:")
                for err in schema_errors:
                    print(f"    - {err}")
                schema_errors_total.extend(schema_errors)

            s = score_site(full, gt["expected"])
            if best_score is None or s["field_score"] > best_score["field_score"]:
                best_score = s

        if best_score:
            tick = lambda b: "pass" if b else "fail"
            print(f"  Best matching site:")
            print(f"    Name:      [{tick(best_score['name_match'])}] extracted='{best_score['name_extracted']}'")
            print(f"    Status:    [{tick(best_score['status_match'])}] extracted='{best_score['status_extracted']}'")
            print(f"    Materials: [{tick(best_score['materials_match'])}] extracted={best_score['materials_extracted']}")
            print(f"    Field score: {best_score['field_score'] * 100:.0f}%")
            total_field_scores.append(best_score["field_score"])

    print_separator()
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if total_field_scores:
        overall = sum(total_field_scores) / len(total_field_scores)
        print(f"  Cases evaluated:  {len(total_field_scores)}/{len(ground_truth)}")
        print(f"  Overall precision: {overall * 100:.1f}%")
        for i, s in enumerate(total_field_scores):
            print(f"    Case {i+1}: {s*100:.0f}%")
    else:
        print("  No cases completed successfully.")

    if schema_errors_total:
        print(f"\n  Schema validation errors: {len(schema_errors_total)}")
        for err in schema_errors_total[:5]:
            print(f"    - {err}")
    else:
        print("\n  Schema validation: ALL PASS")

    print()
    sys.exit(0 if total_field_scores and sum(total_field_scores) / len(total_field_scores) >= 0.5 else 1)


if __name__ == "__main__":
    main()
