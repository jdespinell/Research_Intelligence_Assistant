"""
Tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Research Intelligence Assistant"


def test_corpus_stats():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "num_papers" in data
    assert "num_chunks" in data
    assert isinstance(data["num_papers"], int)
    assert isinstance(data["num_chunks"], int)


def test_list_papers():
    response = client.get("/api/v1/papers")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_query_empty_question():
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 400
