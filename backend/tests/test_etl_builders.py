from datetime import date

import pandas as pd

from app.etl.builders import (
    build_player_advanced_stats_from_row,
    build_player_shot_from_row,
)
from app.etl.backfill import (
    _basketball_reference_season_year,
    _normalize_advanced_stats_dataframe,
)


def test_build_player_shot_from_row_maps_nba_api_columns():
    row = pd.Series(
        {
            "GAME_ID": "0022300001",
            "GAME_EVENT_ID": 42,
            "PERIOD": 2,
            "MINUTES_REMAINING": 8,
            "SECONDS_REMAINING": 17,
            "EVENT_TYPE": "Made Shot",
            "ACTION_TYPE": "Jump Shot",
            "SHOT_TYPE": "2PT Field Goal",
            "SHOT_ZONE_BASIC": "Mid-Range",
            "SHOT_ZONE_AREA": "Right Side(R)",
            "SHOT_ZONE_RANGE": "16-24 ft.",
            "SHOT_DISTANCE": 18,
            "LOC_X": 120,
            "LOC_Y": 80,
            "SHOT_ATTEMPTED_FLAG": 1,
            "SHOT_MADE_FLAG": 1,
            "GAME_DATE": "2023-10-24",
        }
    )

    shot = build_player_shot_from_row(row, player_id=23, season_id=7)

    assert shot.player_id == 23
    assert shot.season_id == 7
    assert shot.game_id == "0022300001"
    assert shot.game_event_id == 42
    assert shot.period == 2
    assert shot.minutes_remaining == 8
    assert shot.seconds_remaining == 17
    assert shot.event_type == "Made Shot"
    assert shot.action_type == "Jump Shot"
    assert shot.shot_zone_basic == "Mid-Range"
    assert shot.shot_zone_area == "Right Side(R)"
    assert shot.shot_zone_range == "16-24 ft."
    assert shot.shot_distance == 18.0
    assert shot.loc_x == 120.0
    assert shot.loc_y == 80.0
    assert shot.shot_attempted_flag is True
    assert shot.shot_made_flag is True
    assert shot.game_date == date(2023, 10, 24)


def test_build_player_shot_from_row_handles_missing_optional_values():
    row = pd.Series(
        {
            "GAME_ID": None,
            "GAME_EVENT_ID": None,
            "SHOT_ATTEMPTED_FLAG": None,
            "SHOT_MADE_FLAG": None,
            "GAME_DATE": "not-a-date",
        }
    )

    shot = build_player_shot_from_row(row, player_id=11, season_id=3)

    assert shot.game_id is None
    assert shot.game_event_id is None
    assert shot.period == 1
    assert shot.shot_attempted_flag is False
    assert shot.shot_made_flag is False
    assert shot.game_date is None


def test_build_player_advanced_stats_from_row_maps_basketball_reference_columns():
    row = pd.Series(
        {
            "PER": "24.8",
            "TS%": ".625",
            "FTr": ".401",
            "ORB%": "3.7",
            "DRB%": "19.8",
            "TRB%": "11.9",
            "AST%": "36.2",
            "STL%": "1.7",
            "BLK%": "1.5",
            "TOV%": "12.9",
            "USG%": "31.4",
            "OWS": "8.1",
            "DWS": "3.2",
            "WS": "11.3",
            "WS/48": ".210",
            "OBPM": "6.9",
            "DBPM": "1.6",
            "BPM": "8.5",
            "VORP": "6.4",
        }
    )

    stats = build_player_advanced_stats_from_row(row, player_id=23, season_id=7)

    assert stats.player_id == 23
    assert stats.season_id == 7
    assert stats.per == 24.8
    assert stats.ts_pct == 0.625
    assert stats.ftr == 0.401
    assert stats.usg_pct == 31.4
    assert stats.ws == 11.3
    assert stats.bpm == 8.5
    assert stats.vorp == 6.4


def test_basketball_reference_season_year_uses_season_ending_year():
    assert _basketball_reference_season_year("2023-24") == 2024
    assert _basketball_reference_season_year("1999-00") == 2000


def test_normalize_advanced_stats_dataframe_removes_repeated_headers_and_blank_players():
    raw = pd.DataFrame(
        [
            {"Player": "Player", "PER": "PER", "VORP": "VORP"},
            {"Player": "Real Player", "PER": "21.0", "VORP": "3.5"},
            {"Player": None, "PER": "18.0", "VORP": "2.1"},
        ]
    )

    normalized = _normalize_advanced_stats_dataframe(raw)

    assert list(normalized["Player"]) == ["Real Player"]
    assert normalized.iloc[0]["PER"] == "21.0"
