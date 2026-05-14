"""
Historical backfill script for NBA data ingestion.

Usage:
    cd backend && python -m app.etl.backfill --seasons 2023-24 2024-25 --game-logs
    cd backend && python -m app.etl.backfill --seasons 2023-24 --shots --limit-players 10
"""
import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Set, Tuple

import pandas as pd
from sqlmodel import Session, select
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Add backend root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.deps import get_db_context
from app.models import Player, Team, Season, PlayerGameLog, PlayerShot
from app.etl.builders import build_player_advanced_stats_from_row, build_player_shot_from_row
from app.repositories.historical import HistoricalRepository
from app.repositories.team import TeamRepository

# nba_api imports
from nba_api.stats.endpoints import playergamelog, shotchartdetail
from nba_api.stats.static import players as static_players


REQUEST_DELAY = 0.6  # seconds between nba_api requests
BATCH_SIZE = 500
BASKETBALL_REFERENCE_ADVANCED_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"


def _parse_matchup_team_abbr(matchup: str) -> Optional[str]:
    """Extract team abbreviation from matchup string like 'LAL @ BOS' or 'LAL vs. BOS'."""
    if not matchup:
        return None
    parts = matchup.split()
    return parts[0] if parts else None


def _get_or_create_season(session: Session, season_label: str) -> Season:
    stmt = select(Season).where(Season.season_label == season_label)
    result = session.exec(stmt).first()
    if result:
        return result
    season = Season(season_label=season_label)
    session.add(season)
    session.commit()
    session.refresh(season)
    return season


def _resolve_team_id(session: Session, abbr: Optional[str]) -> Optional[int]:
    if not abbr:
        return None
    repo = TeamRepository(session)
    team = repo.get_by_abbreviation(abbr)
    return team.id if team else None


def _basketball_reference_season_year(season_label: str) -> int:
    """Convert NBA season label (2023-24) into Basketball Reference ending year (2024)."""
    start_year, end_suffix = season_label.split("-")
    century = int(start_year[:2]) * 100
    end_year = century + int(end_suffix)
    if end_year < int(start_year):
        end_year += 100
    return end_year


def _normalize_advanced_stats_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated Basketball Reference header rows and blank players."""
    normalized = df.copy()
    normalized = normalized[normalized["Player"].notna()]
    normalized = normalized[normalized["Player"] != "Player"]
    return normalized.reset_index(drop=True)


def _fetch_basketball_reference_advanced_stats(season_label: str) -> pd.DataFrame:
    year = _basketball_reference_season_year(season_label)
    url = BASKETBALL_REFERENCE_ADVANCED_URL.format(year=year)
    tables = pd.read_html(url)
    if not tables:
        return pd.DataFrame()
    return _normalize_advanced_stats_dataframe(tables[0])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_player_game_logs(player_id: int, season: str) -> Optional[pd.DataFrame]:
    """Fetch game logs for a player in a season. Returns None if no data."""
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = gamelog.get_data_frames()[0]
        if df.empty:
            return None
        return df
    except Exception as e:
        # Some players may not have data for the season
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise  # Let tenacity retry
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_player_shots(player_id: int, season: str, team_id: int = 0) -> Optional[pd.DataFrame]:
    """Fetch shot chart detail for a player in a season."""
    try:
        sc = shotchartdetail.ShotChartDetail(
            team_id=team_id,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
        )
        df = sc.get_data_frames()[0]
        if df.empty:
            return None
        return df
    except Exception as e:
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            raise
        return None


def backfill_game_logs(
    seasons: List[str],
    limit_players: Optional[int] = None,
    skip_existing: bool = True,
) -> dict:
    """Backfill game logs for all players across specified seasons."""
    with get_db_context() as session:
        # Get players from DB
        players = session.exec(select(Player)).all()
        if limit_players:
            players = players[:limit_players]

        # Track already processed (player_id, game_id) to skip duplicates
        existing_keys: Set[Tuple[int, str]] = set()
        if skip_existing:
            rows = session.exec(select(PlayerGameLog.player_id, PlayerGameLog.game_id)).all()
            existing_keys = {(r[0], r[1]) for r in rows}

        summary = {
            "seasons": seasons,
            "players_processed": 0,
            "games_inserted": 0,
            "games_skipped": 0,
            "errors": 0,
        }

        for season_label in seasons:
            season = _get_or_create_season(session, season_label)
            print(f"\n=== Season {season_label} ===")

            for player in tqdm(players, desc=f"Game logs {season_label}"):
                try:
                    df = _fetch_player_game_logs(player.id, season_label)
                    time.sleep(REQUEST_DELAY)

                    if df is None:
                        continue

                    batch = []
                    for _, row in df.iterrows():
                        game_id = str(row.get("Game_ID", ""))
                        if not game_id:
                            continue

                        key = (player.id, game_id)
                        if key in existing_keys:
                            summary["games_skipped"] += 1
                            continue

                        matchup = str(row.get("MATCHUP", ""))
                        team_abbr = _parse_matchup_team_abbr(matchup)
                        team_id = _resolve_team_id(session, team_abbr) or 0

                        game_date_raw = row.get("GAME_DATE")
                        game_date = None
                        if game_date_raw and pd.notna(game_date_raw):
                            try:
                                game_date = datetime.strptime(str(game_date_raw), "%b %d, %Y").date()
                            except ValueError:
                                try:
                                    game_date = datetime.strptime(str(game_date_raw), "%Y-%m-%d").date()
                                except ValueError:
                                    pass

                        log = PlayerGameLog(
                            player_id=player.id,
                            team_id=team_id,
                            season_id=season.id,
                            game_id=game_id,
                            game_date=game_date,
                            matchup=matchup,
                            wl=str(row.get("WL", "")) if pd.notna(row.get("WL")) else None,
                            min=int(row.get("MIN", 0)) if pd.notna(row.get("MIN")) else 0,
                            fgm=int(row.get("FGM", 0)) if pd.notna(row.get("FGM")) else 0,
                            fga=int(row.get("FGA", 0)) if pd.notna(row.get("FGA")) else 0,
                            fg_pct=float(row.get("FG_PCT", 0)) if pd.notna(row.get("FG_PCT")) else 0.0,
                            fg3m=int(row.get("FG3M", 0)) if pd.notna(row.get("FG3M")) else 0,
                            fg3a=int(row.get("FG3A", 0)) if pd.notna(row.get("FG3A")) else 0,
                            fg3_pct=float(row.get("FG3_PCT", 0)) if pd.notna(row.get("FG3_PCT")) else 0.0,
                            ftm=int(row.get("FTM", 0)) if pd.notna(row.get("FTM")) else 0,
                            fta=int(row.get("FTA", 0)) if pd.notna(row.get("FTA")) else 0,
                            ft_pct=float(row.get("FT_PCT", 0)) if pd.notna(row.get("FT_PCT")) else 0.0,
                            oreb=int(row.get("OREB", 0)) if pd.notna(row.get("OREB")) else 0,
                            dreb=int(row.get("DREB", 0)) if pd.notna(row.get("DREB")) else 0,
                            reb=int(row.get("REB", 0)) if pd.notna(row.get("REB")) else 0,
                            ast=int(row.get("AST", 0)) if pd.notna(row.get("AST")) else 0,
                            stl=int(row.get("STL", 0)) if pd.notna(row.get("STL")) else 0,
                            blk=int(row.get("BLK", 0)) if pd.notna(row.get("BLK")) else 0,
                            tov=int(row.get("TOV", 0)) if pd.notna(row.get("TOV")) else 0,
                            pf=int(row.get("PF", 0)) if pd.notna(row.get("PF")) else 0,
                            pts=int(row.get("PTS", 0)) if pd.notna(row.get("PTS")) else 0,
                            plus_minus=int(row.get("PLUS_MINUS", 0)) if pd.notna(row.get("PLUS_MINUS")) else 0,
                        )
                        batch.append(log)
                        existing_keys.add(key)

                        if len(batch) >= BATCH_SIZE:
                            session.add_all(batch)
                            session.commit()
                            summary["games_inserted"] += len(batch)
                            batch = []

                    if batch:
                        session.add_all(batch)
                        session.commit()
                        summary["games_inserted"] += len(batch)

                    summary["players_processed"] += 1

                except Exception as e:
                    summary["errors"] += 1
                    print(f"Error processing player {player.full_name} ({player.id}): {e}")
                    session.rollback()
                    continue

        return summary


def backfill_shots(
    seasons: List[str],
    limit_players: Optional[int] = None,
    skip_existing: bool = True,
) -> dict:
    """Backfill shot charts for all players across specified seasons.
    WARNING: This is very slow due to the volume of data."""
    with get_db_context() as session:
        players = session.exec(select(Player)).all()
        if limit_players:
            players = players[:limit_players]

        existing_keys: Set[Tuple[int, str, int]] = set()
        if skip_existing:
            rows = session.exec(
                select(PlayerShot.player_id, PlayerShot.game_id, PlayerShot.game_event_id)
            ).all()
            existing_keys = {(r[0], r[1], r[2]) for r in rows if r[1] and r[2] is not None}

        summary = {
            "seasons": seasons,
            "players_processed": 0,
            "shots_inserted": 0,
            "shots_skipped": 0,
            "errors": 0,
        }

        for season_label in seasons:
            season = _get_or_create_season(session, season_label)
            print(f"\n=== Shot Charts {season_label} ===")

            for player in tqdm(players, desc=f"Shots {season_label}"):
                try:
                    df = _fetch_player_shots(player.id, season_label)
                    time.sleep(REQUEST_DELAY)

                    if df is None:
                        continue

                    batch = []
                    for _, row in df.iterrows():
                        game_id = str(row.get("GAME_ID", "")) if pd.notna(row.get("GAME_ID")) else None
                        game_event_id = int(row.get("GAME_EVENT_ID", 0)) if pd.notna(row.get("GAME_EVENT_ID")) else None

                        if game_id and game_event_id is not None:
                            key = (player.id, game_id, game_event_id)
                            if key in existing_keys:
                                summary["shots_skipped"] += 1
                                continue
                            existing_keys.add(key)

                        shot = build_player_shot_from_row(row, player_id=player.id, season_id=season.id)
                        batch.append(shot)

                        if len(batch) >= BATCH_SIZE:
                            session.add_all(batch)
                            session.commit()
                            summary["shots_inserted"] += len(batch)
                            batch = []

                    if batch:
                        session.add_all(batch)
                        session.commit()
                        summary["shots_inserted"] += len(batch)

                    summary["players_processed"] += 1

                except Exception as e:
                    summary["errors"] += 1
                    print(f"Error processing shots for {player.full_name}: {e}")
                    session.rollback()
                    continue

        return summary


def backfill_advanced_stats(
    seasons: List[str],
    limit_players: Optional[int] = None,
) -> dict:
    """Backfill Basketball Reference advanced stats for known players."""
    with get_db_context() as session:
        repo = HistoricalRepository(session)
        player_by_name = {player.full_name: player for player in session.exec(select(Player)).all()}

        summary = {
            "seasons": seasons,
            "players_processed": 0,
            "advanced_stats_upserted": 0,
            "players_skipped": 0,
            "errors": 0,
        }

        for season_label in seasons:
            season = _get_or_create_season(session, season_label)
            try:
                df = _fetch_basketball_reference_advanced_stats(season_label)
            except Exception as e:
                summary["errors"] += 1
                print(f"Error fetching advanced stats for {season_label}: {e}")
                continue

            if limit_players:
                df = df.head(limit_players)

            for _, row in df.iterrows():
                player_name = str(row.get("Player", ""))
                player = player_by_name.get(player_name)
                if not player:
                    summary["players_skipped"] += 1
                    continue

                try:
                    stats = build_player_advanced_stats_from_row(row, player.id, season.id)
                    repo.save_advanced_stats(stats)
                    summary["players_processed"] += 1
                    summary["advanced_stats_upserted"] += 1
                except Exception as e:
                    summary["errors"] += 1
                    print(f"Error processing advanced stats for {player_name}: {e}")
                    session.rollback()

        return summary


def main():
    parser = argparse.ArgumentParser(description="NBA historical data backfill")
    parser.add_argument("--seasons", nargs="+", default=["2023-24"], help="Seasons to backfill")
    parser.add_argument("--game-logs", action="store_true", help="Backfill game logs")
    parser.add_argument("--shots", action="store_true", help="Backfill shot charts (slow)")
    parser.add_argument("--advanced-stats", action="store_true", help="Backfill Basketball Reference advanced stats")
    parser.add_argument("--limit-players", type=int, default=None, help="Limit number of players")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-process existing data")
    args = parser.parse_args()

    skip_existing = not args.no_skip_existing

    if not args.game_logs and not args.shots and not args.advanced_stats:
        print("Use --game-logs, --shots and/or --advanced-stats")
        return

    if args.game_logs:
        print("Starting game logs backfill...")
        result = backfill_game_logs(args.seasons, args.limit_players, skip_existing)
        print("\nGame logs result:", result)

    if args.shots:
        print("Starting shot charts backfill...")
        result = backfill_shots(args.seasons, args.limit_players, skip_existing)
        print("\nShots result:", result)

    if args.advanced_stats:
        print("Starting advanced stats backfill...")
        result = backfill_advanced_stats(args.seasons, args.limit_players)
        print("\nAdvanced stats result:", result)


if __name__ == "__main__":
    main()
