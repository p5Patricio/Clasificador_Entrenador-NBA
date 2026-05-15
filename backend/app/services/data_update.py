from pathlib import Path
from typing import Any, Dict, Optional

from sqlmodel import Session

from app.etl.pipeline import ETLPipeline


class DataUpdateService:
    def __init__(self, session: Session, cache=None):
        self.session = session
        self.cache = cache

    def _default_filepath(self, season: str, season_type: str, min_minutes: int) -> str:
        normalized_season_type = season_type.replace(" ", "_")
        filename = f"nba_active_player_stats_{season}_{normalized_season_type}_{min_minutes}min.xlsx"
        return str(Path(__file__).resolve().parents[3] / filename)

    def run_update(
        self,
        season: str,
        season_type: str = "Regular Season",
        min_minutes: int = 100,
        filepath: Optional[str] = None,
    ) -> Dict[str, Any]:
        target_file = filepath or self._default_filepath(season, season_type, min_minutes)
        result = ETLPipeline(self.session).run(target_file, season)
        if self.cache:
            self.cache.delete_prefix("historical:")
        return result
