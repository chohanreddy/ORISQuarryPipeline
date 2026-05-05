# Celery worker - picks up jobs from Redis and runs the full pipeline
from __future__ import annotations
import logging
import os
import time
import uuid
from datetime import datetime, timezone

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("quarry_worker", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@celery_app.task(bind=True, name="worker.process_quarry_job", max_retries=0)
def process_quarry_job(self, job_id: str):
    from models.db import get_db, Job, Site
    from pipeline.discovery import discover_quarries
    from pipeline.scraper import fetch_page, fetch_osm_page_text, content_hash
    from pipeline.extractor import extract_from_source, extract_from_osm_tags
    from pipeline.reconciler import reconcile
    from pipeline.discovery import _reverse_geocode_city

    db = get_db()
    job = db.query(Job).filter_by(id=job_id).first()
    if not job:
        logger.error("Job %s not found", job_id)
        return

    job.status = "running"
    job.updated_at = datetime.utcnow()
    db.commit()

    input_data = job.input_data
    lat = input_data["latitude"]
    lon = input_data["longitude"]
    radius_km = input_data["radius_km"]
    max_cost = input_data.get("max_usd_cost")

    job_start = time.time()
    total_cost = 0.0

    # dont let one job go crazy and process hundreds of sites
    MAX_CANDIDATES = 30

    try:
        logger.info("Starting discovery for job %s (%.4f, %.4f, %skm)", job_id, lat, lon, radius_km)
        candidates = discover_quarries(lat, lon, radius_km)

        # resolve city once for the whole job, reused in location verification per candidate
        expected_city = _reverse_geocode_city(lat, lon)
        logger.info("Expected city for location verification: %s", expected_city)

        if len(candidates) > MAX_CANDIDATES:
            logger.info("Capping %d candidates to %d", len(candidates), MAX_CANDIDATES)
            candidates = candidates[:MAX_CANDIDATES]

        job.progress = 10
        job.updated_at = datetime.utcnow()
        db.commit()

        if not candidates:
            job.status = "completed"
            job.progress = 100
            job.updated_at = datetime.utcnow()
            db.commit()
            logger.info("No candidates found for job %s", job_id)
            return

        site_count = 0
        for i, candidate in enumerate(candidates):
            # if user set a cost cap, stop once we hit it
            if max_cost and total_cost >= max_cost:
                logger.info("Reached cost limit $%.4f, stopping early", max_cost)
                break

            candidate_start = time.time()
            fetched_at = _now_iso()
            source_hashes: dict = {}
            scraped_extractions = []
            model_calls = []

            osm_text = fetch_osm_page_text(candidate.osm_tags)
            osm_source_id = f"src_osm_{i}"
            source_hashes[osm_source_id] = content_hash(osm_text)
            osm_extraction = extract_from_osm_tags(candidate.osm_tags, osm_source_id)

            url_limit = min(len(candidate.candidate_urls), 3)
            for j, url in enumerate(candidate.candidate_urls[:url_limit]):
                src_id = f"src_{i}_{j}"
                result = fetch_page(url)
                if result is None:
                    continue
                text, chash, last_modified = result
                source_hashes[src_id] = chash

                tier = candidate.trust_tier if j == 0 else "directory"
                extraction, model_call = extract_from_source(text, src_id, url, last_modified=last_modified)
                scraped_extractions.append((extraction, src_id, url, tier))
                model_calls.append(model_call)
                total_cost += model_call.get("usd_cost", 0.0)

            latency_ms = int((time.time() - candidate_start) * 1000)

            record = reconcile(
                candidate=candidate,
                scraped_extractions=scraped_extractions,
                osm_extraction=osm_extraction,
                osm_source_id=osm_source_id,
                model_calls=model_calls,
                input_data=input_data,
                latency_ms=latency_ms,
                fetched_at=fetched_at,
                source_content_hashes=source_hashes,
                expected_city=expected_city,
            )

            # insert or update the site record
            existing = db.query(Site).filter_by(site_id=record.site_id).first()
            if existing:
                existing.job_id = job_id
                existing.data = record.model_dump()
                existing.official_name = record.extraction.official_name.value if record.extraction.official_name else None
                existing.operational_status = record.extraction.operational_status.value if record.extraction.operational_status else None
                existing.confidence = record.extraction.official_name.confidence if record.extraction.official_name else 0.0
                existing.latitude = candidate.latitude
                existing.longitude = candidate.longitude
            else:
                site = Site(
                    id=str(uuid.uuid4()),
                    site_id=record.site_id,
                    job_id=job_id,
                    data=record.model_dump(),
                    official_name=record.extraction.official_name.value if record.extraction.official_name else None,
                    operational_status=record.extraction.operational_status.value if record.extraction.operational_status else None,
                    confidence=record.extraction.official_name.confidence if record.extraction.official_name else 0.0,
                    latitude=candidate.latitude,
                    longitude=candidate.longitude,
                )
                db.add(site)

            site_count += 1
            db.commit()

            progress = 10 + int((i + 1) / len(candidates) * 85)
            job.progress = progress
            job.result_count = site_count
            job.updated_at = datetime.utcnow()
            db.commit()
            logger.info("Processed candidate %d/%d: %s", i + 1, len(candidates), candidate.name)

        job.status = "completed"
        job.progress = 100
        job.result_count = site_count
        job.updated_at = datetime.utcnow()
        db.commit()
        logger.info("Job %s completed. %d sites extracted. Total cost: $%.4f", job_id, site_count, total_cost)

    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        job.status = "failed"
        job.error = str(exc)
        job.updated_at = datetime.utcnow()
        db.commit()
        raise
    finally:
        db.close()
