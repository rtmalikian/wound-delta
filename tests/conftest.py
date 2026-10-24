import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AI_BACKEND", "stub")

from app import gateway  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_ledger():
    gateway.reset_ledger()
    yield
    gateway.reset_ledger()
