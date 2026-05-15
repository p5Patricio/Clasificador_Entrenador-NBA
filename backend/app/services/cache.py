import json
from typing import Any, Optional


class CacheService:
    def __init__(self, client=None, default_ttl_seconds: int = 300):
        self.client = client
        self.default_ttl_seconds = default_ttl_seconds

    def get_json(self, key: str) -> Optional[Any]:
        if self.client is None:
            return None
        try:
            value = self.client.get(key)
            if value is None:
                return None
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return json.loads(value)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if self.client is None:
            return
        try:
            self.client.setex(
                key,
                ttl_seconds or self.default_ttl_seconds,
                json.dumps(value),
            )
        except Exception:
            return

    def delete_prefix(self, prefix: str) -> None:
        if self.client is None:
            return
        try:
            keys = list(self.client.scan_iter(match=f"{prefix}*"))
            if keys:
                self.client.delete(*keys)
        except Exception:
            return
