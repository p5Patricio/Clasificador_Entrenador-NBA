from app.services.cache import CacheService


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.deleted = []

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl_seconds, value):
        self.values[key] = value
        self.values[f"{key}:ttl"] = ttl_seconds

    def scan_iter(self, match):
        prefix = match.rstrip("*")
        return [key for key in self.values if key.startswith(prefix) and not key.endswith(":ttl")]

    def delete(self, *keys):
        self.deleted.extend(keys)
        for key in keys:
            self.values.pop(key, None)


def test_cache_service_round_trips_json_values_and_deletes_prefix():
    fake = FakeRedis()
    cache = CacheService(client=fake, default_ttl_seconds=60)

    cache.set_json("historical:player:1", {"pts": 27})

    assert cache.get_json("historical:player:1") == {"pts": 27}
    assert fake.values["historical:player:1:ttl"] == 60

    cache.delete_prefix("historical:")

    assert fake.deleted == ["historical:player:1"]
    assert cache.get_json("historical:player:1") is None


def test_cache_service_noops_when_client_is_missing():
    cache = CacheService(client=None)

    cache.set_json("x", {"value": 1})
    cache.delete_prefix("x")

    assert cache.get_json("x") is None
