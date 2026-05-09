from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "NBA Analytics Platform API" in data["message"]


def test_players_empty(client: TestClient):
    response = client.get("/api/v1/players")
    assert response.status_code == 200
    assert response.json() == []


def test_teams_empty(client: TestClient):
    response = client.get("/api/v1/teams")
    assert response.status_code == 200
    assert response.json() == []
