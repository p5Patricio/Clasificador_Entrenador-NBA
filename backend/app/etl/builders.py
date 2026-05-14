from datetime import datetime
from typing import Any, Optional

import pandas as pd

from app.models import PlayerAdvancedStats, PlayerShot


def _is_present(value: Any) -> bool:
    return value is not None and pd.notna(value)


def _str_or_none(value: Any) -> Optional[str]:
    return str(value) if _is_present(value) else None


def _int_or_default(value: Any, default: int = 0) -> int:
    return int(value) if _is_present(value) else default


def _float_or_default(value: Any, default: float = 0.0) -> float:
    return float(value) if _is_present(value) else default


def _float_or_none(value: Any) -> Optional[float]:
    return float(value) if _is_present(value) else None


def _parse_iso_date(value: Any):
    if not _is_present(value):
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def build_player_shot_from_row(row: pd.Series, player_id: int, season_id: int) -> PlayerShot:
    return PlayerShot(
        player_id=player_id,
        season_id=season_id,
        game_id=_str_or_none(row.get("GAME_ID")),
        game_event_id=_int_or_default(row.get("GAME_EVENT_ID")) if _is_present(row.get("GAME_EVENT_ID")) else None,
        period=_int_or_default(row.get("PERIOD"), 1),
        minutes_remaining=_int_or_default(row.get("MINUTES_REMAINING")),
        seconds_remaining=_int_or_default(row.get("SECONDS_REMAINING")),
        event_type=_str_or_none(row.get("EVENT_TYPE")),
        action_type=_str_or_none(row.get("ACTION_TYPE")),
        shot_type=_str_or_none(row.get("SHOT_TYPE")),
        shot_zone_basic=_str_or_none(row.get("SHOT_ZONE_BASIC")),
        shot_zone_area=_str_or_none(row.get("SHOT_ZONE_AREA")),
        shot_zone_range=_str_or_none(row.get("SHOT_ZONE_RANGE")),
        shot_distance=_float_or_default(row.get("SHOT_DISTANCE")),
        loc_x=_float_or_default(row.get("LOC_X")),
        loc_y=_float_or_default(row.get("LOC_Y")),
        shot_attempted_flag=bool(row.get("SHOT_ATTEMPTED_FLAG")) if _is_present(row.get("SHOT_ATTEMPTED_FLAG")) else False,
        shot_made_flag=bool(row.get("SHOT_MADE_FLAG")) if _is_present(row.get("SHOT_MADE_FLAG")) else False,
        game_date=_parse_iso_date(row.get("GAME_DATE")),
    )


def build_player_advanced_stats_from_row(
    row: pd.Series,
    player_id: int,
    season_id: int,
) -> PlayerAdvancedStats:
    return PlayerAdvancedStats(
        player_id=player_id,
        season_id=season_id,
        per=_float_or_none(row.get("PER")),
        ts_pct=_float_or_none(row.get("TS%")),
        ftr=_float_or_none(row.get("FTr")),
        orb_pct=_float_or_none(row.get("ORB%")),
        drb_pct=_float_or_none(row.get("DRB%")),
        trb_pct=_float_or_none(row.get("TRB%")),
        ast_pct=_float_or_none(row.get("AST%")),
        stl_pct=_float_or_none(row.get("STL%")),
        blk_pct=_float_or_none(row.get("BLK%")),
        tov_pct=_float_or_none(row.get("TOV%")),
        usg_pct=_float_or_none(row.get("USG%")),
        ows=_float_or_none(row.get("OWS")),
        dws=_float_or_none(row.get("DWS")),
        ws=_float_or_none(row.get("WS")),
        ws_per_48=_float_or_none(row.get("WS/48")),
        obpm=_float_or_none(row.get("OBPM")),
        dbpm=_float_or_none(row.get("DBPM")),
        bpm=_float_or_none(row.get("BPM")),
        vorp=_float_or_none(row.get("VORP")),
    )
