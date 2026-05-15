from app.services import data_update as data_update_module
from app.services.data_update import DataUpdateService


class FakePipeline:
    def __init__(self, session):
        self.session = session

    def run(self, filepath, season):
        return {
            "season": season,
            "file": filepath,
            "rows_processed": 1,
            "inserted": 1,
            "updated": 0,
            "failed": 0,
        }


class FakeCache:
    def __init__(self):
        self.deleted_prefixes = []

    def delete_prefix(self, prefix):
        self.deleted_prefixes.append(prefix)


def test_data_update_service_invalidates_historical_cache(monkeypatch, db_session):
    cache = FakeCache()
    monkeypatch.setattr(data_update_module, "ETLPipeline", FakePipeline)

    result = DataUpdateService(db_session, cache=cache).run_update(
        season="2024-25",
        filepath="stats.xlsx",
    )

    assert result["inserted"] == 1
    assert cache.deleted_prefixes == ["historical:"]
