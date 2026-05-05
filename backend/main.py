# FastAPI app - handles all the REST endpoints for jobs and sites
from __future__ import annotations
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text as sa_text

from models.db import create_tables, get_db, Job, Site
from models.schema import JobRequest, JobResponse, JobStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ORIS Quarry Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    create_tables()
    logger.info("Database tables ready")


# --- jobs ---


@app.post("/api/jobs", response_model=JobResponse, status_code=202)
def create_job(req: JobRequest):
    from worker import process_quarry_job

    job_id = str(uuid.uuid4())
    db = get_db()
    try:
        job = Job(
            id=job_id,
            status="pending",
            progress=0,
            input_data={
                "latitude": req.latitude,
                "longitude": req.longitude,
                "radius_km": req.radius_km,
                "max_usd_cost": req.max_usd_cost,
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    process_quarry_job.delay(job_id)
    logger.info("Submitted job %s", job_id)
    return JobResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    db = get_db()
    try:
        job = db.query(Job).filter_by(id=job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job.id,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at.isoformat() + "Z",
            "updated_at": job.updated_at.isoformat() + "Z",
            "result_count": job.result_count or 0,
            "error": job.error,
        }
    finally:
        db.close()


# --- sites ---


@app.get("/api/sites")
def list_sites(
    q: Optional[str] = Query(None, description="Filter by name (case-insensitive)"),
    status: Optional[str] = Query(None, description="Filter by operational_status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    db = get_db()
    try:
        query = db.query(Site)
        if q:
            query = query.filter(Site.official_name.ilike(f"%{q}%"))
        if status:
            query = query.filter(Site.operational_status == status)

        total = query.count()
        sites = query.order_by(Site.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

        def _materials(site) -> list:
            try:
                mats = site.data.get("extraction", {}).get("materials_produced", [])
                return [m["value"] for m in mats if m.get("value")]
            except Exception:
                return []

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [
                {
                    "id": s.id,
                    "site_id": s.site_id,
                    "job_id": s.job_id,
                    "official_name": s.official_name,
                    "operational_status": s.operational_status,
                    "confidence": s.confidence,
                    "materials": _materials(s),
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                    "created_at": s.created_at.isoformat() + "Z",
                }
                for s in sites
            ],
        }
    finally:
        db.close()


@app.get("/api/sites/{site_id}")
def get_site(site_id: str):
    db = get_db()
    try:
        site = db.query(Site).filter_by(id=site_id).first()
        if not site:
            # also try by the site_id field, not just DB primary key
            site = db.query(Site).filter_by(site_id=site_id).first()
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")
        return site.data
    finally:
        db.close()


# --- health check ---


@app.get("/api/health")
def health():
    import redis as redis_lib

    db = get_db()
    try:
        db.execute(sa_text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    finally:
        db.close()

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        r = redis_lib.from_url(redis_url)
        r.ping()
        redis_ok = True
        queue_depth = r.llen("celery")
    except Exception:
        redis_ok = False
        queue_depth = -1

    # count failures to get a rough error rate
    db2 = get_db()
    try:
        total_jobs = db2.query(Job).count()
        failed_jobs = db2.query(Job).filter_by(status="failed").count()
        error_rate = (failed_jobs / total_jobs) if total_jobs > 0 else 0.0
    except Exception:
        error_rate = 0.0
        total_jobs = 0
    finally:
        db2.close()

    try:
        from worker import celery_app
        ping = celery_app.control.inspect(timeout=0.5).ping()
        worker_count = len(ping) if ping else 0
    except Exception:
        worker_count = -1

    return {
        "status": "ok" if (db_ok and redis_ok) else "degraded",
        "database": "ok" if db_ok else "error",
        "redis": "ok" if redis_ok else "error",
        "queue_depth": queue_depth,
        "worker_count": worker_count,
        "total_jobs": total_jobs,
        "error_rate": round(error_rate, 4),
    }
