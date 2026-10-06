from fastapi.testclient import TestClient

from gaokao_agent.api import app


def test_health_endpoint_reports_index_mode() -> None:
    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["index_ready"] is True
    expected_mode = "hybrid_bm25_cosine" if payload["vector_enabled"] else "bm25"
    assert payload["retrieval_mode"] == expected_mode


def test_built_vue_frontend_is_served() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "高考志愿助手" in response.text
