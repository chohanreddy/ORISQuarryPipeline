# shared fixtures - patches out DB and Celery so tests run without any infra
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

# set dummy env vars before any imports so SQLAlchemy doesnt blow up on missing DB
os.environ.setdefault("POSTGRES_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


@pytest.fixture
def app_client():
    # FastAPI test client with DB and Celery completely stubbed out
    from fastapi.testclient import TestClient
    with patch("main.create_tables"):
        with patch("worker.process_quarry_job"):
            import main as app_module
            with TestClient(app_module.app, raise_server_exceptions=False) as client:
                yield client
