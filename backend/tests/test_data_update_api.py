from app.api.deps import get_data_update_service
from app.main import app


class FakeDataUpdateService:
    def run_update(self, season: str, season_type: str, min_minutes: int, filepath: str | None = None):
        assert season == "2024-25"
        assert season_type == "Regular Season"
        assert min_minutes == 100
        assert filepath == "nba_active_player_stats_2024-25_Regular_Season_100min.xlsx"
        return {
            "season": season,
            "file": filepath,
            "rows_processed": 2,
            "inserted": 1,
            "updated": 1,
            "failed": 0,
        }


def test_data_update_delegates_to_service_and_returns_real_summary(client):
    app.dependency_overrides[get_data_update_service] = lambda: FakeDataUpdateService()

    try:
        response = client.post(
            "/api/v1/data/update",
            json={
                "season": "2024-25",
                "season_type": "Regular Season",
                "min_minutes": 100,
                "filepath": "nba_active_player_stats_2024-25_Regular_Season_100min.xlsx",
            },
        )
    finally:
        app.dependency_overrides.pop(get_data_update_service, None)

    assert response.status_code == 200
    assert response.json() == {
        "season": "2024-25",
        "file": "nba_active_player_stats_2024-25_Regular_Season_100min.xlsx",
        "rows_processed": 2,
        "players_inserted": 1,
        "players_updated": 1,
        "players_failed": 0,
        "etl_status": "completed",
    }
