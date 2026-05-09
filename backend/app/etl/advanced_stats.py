"""
Basketball Reference advanced stats scraper.

Usage:
    cd backend && python -m app.etl.advanced_stats --seasons 2024 2025
"""
import argparse
import os
import sys
import time
from typing import List, Optional

import pandas as pd
from sqlmodel import Session, select
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.deps import get_db_context
from app.models import Player, Season, PlayerAdvancedStats


BR_ADVANCED_URL = "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
REQUEST_DELAY = 3.0  # Be nice to Basketball Reference

# Column mapping from Basketball Reference to our model
COLUMN_MAP = {
    "PER": "per",
    "TS%": "ts_pct",
    "FTr": "ftr",
    "ORB%": "orb_pct",
    "DRB%": "drb_pct",
    "TRB%": "trb_pct",
    "AST%": "ast_pct",
    "STL%": "stl_pct",
    "BLK%": "blk_pct",
    "TOV%": "tov_pct",
    "USG%": "usg_pct",
    "OWS": "ows",
    "DWS": "dws",
    "WS": "ws",
    "WS/48": "ws_per_48",
    "OBPM": "obpm",
    "DBPM": "dbpm",
    "BPM": "bpm",
    "VORP": "vorp",
}


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


def _resolve_player_id(session: Session, name: str) -> Optional[int]:
    """Match Basketball Reference player name to our Player table."""
    # Exact match first
    player = session.exec(select(Player).where(Player.full_name == name)).first()
    if player:
        return player.id
    # Case-insensitive partial match
    players = session.exec(select(Player)).all()
    name_lower = name.lower()
    for p in players:
        if name_lower in p.full_name.lower() or p.full_name.lower() in name_lower:
            return p.id
    return None


def scrape_advanced_stats(year: int) -> Optional[pd.DataFrame]:
    """Scrape advanced stats table from Basketball Reference for a given NBA year.
    
    year: The ending year of the season (e.g., 2024 for 2023-24 season)
    """
    url = BR_ADVANCED_URL.format(year=year)
    try:
        tables = pd.read_html(url)
        # The advanced stats table is usually the first one on the page
        for df in tables:
            if "PER" in df.columns or (isinstance(df.columns, pd.MultiIndex) and "PER" in df.columns.get_level_values(-1)):
                # Handle multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                return df
        return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def backfill_advanced_stats(seasons: List[str]) -> dict:
    """Backfill advanced stats for specified seasons."""
    with get_db_context() as session:
        summary = {
            "seasons": seasons,
            "inserted": 0,
            "updated": 0,
            "not_found": 0,
            "errors": 0,
        }

        for season_label in seasons:
            # Convert "2023-24" to year integer (2024)
            try:
                year = int(season_label.split("-")[0]) + 1
            except (ValueError, IndexError):
                print(f"Invalid season format: {season_label}")
                summary["errors"] += 1
                continue

            print(f"\n=== Scraping advanced stats for {season_label} (year {year}) ===")
            season = _get_or_create_season(session, season_label)

            df = scrape_advanced_stats(year)
            time.sleep(REQUEST_DELAY)

            if df is None or df.empty:
                print(f"No data found for {season_label}")
                continue

            # Clean up dataframe
            # Drop header rows that may repeat in the table
            df = df[df["Rk"].notna()].copy()
            df = df[df["Rk"] != "Rk"].copy()

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {season_label}"):
                try:
                    name = str(row.get("Player", ""))
                    if not name:
                        continue

                    player_id = _resolve_player_id(session, name)
                    if not player_id:
                        summary["not_found"] += 1
                        continue

                    # Check if exists
                    existing = session.exec(
                        select(PlayerAdvancedStats).where(
                            PlayerAdvancedStats.player_id == player_id,
                            PlayerAdvancedStats.season_id == season.id,
                        )
                    ).first()

                    def get_float(col: str) -> Optional[float]:
                        val = row.get(col)
                        if pd.isna(val):
                            return None
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return None

                    stats = PlayerAdvancedStats(
                        player_id=player_id,
                        season_id=season.id,
                        per=get_float("PER"),
                        ts_pct=get_float("TS%"),
                        ftr=get_float("FTr"),
                        orb_pct=get_float("ORB%"),
                        drb_pct=get_float("DRB%"),
                        trb_pct=get_float("TRB%"),
                        ast_pct=get_float("AST%"),
                        stl_pct=get_float("STL%"),
                        blk_pct=get_float("BLK%"),
                        tov_pct=get_float("TOV%"),
                        usg_pct=get_float("USG%"),
                        ows=get_float("OWS"),
                        dws=get_float("DWS"),
                        ws=get_float("WS"),
                        ws_per_48=get_float("WS/48"),
                        obpm=get_float("OBPM"),
                        dbpm=get_float("DBPM"),
                        bpm=get_float("BPM"),
                        vorp=get_float("VORP"),
                    )

                    if existing:
                        for key, value in stats.model_dump(exclude={"id"}).items():
                            setattr(existing, key, value)
                        session.add(existing)
                        summary["updated"] += 1
                    else:
                        session.add(stats)
                        summary["inserted"] += 1

                    session.commit()

                except Exception as e:
                    summary["errors"] += 1
                    print(f"Error processing row: {e}")
                    session.rollback()
                    continue

        return summary


def main():
    parser = argparse.ArgumentParser(description="Basketball Reference advanced stats scraper")
    parser.add_argument("--seasons", nargs="+", required=True, help="Seasons to scrape (e.g., 2023-24 2024-25)")
    args = parser.parse_args()

    print("Starting advanced stats backfill...")
    result = backfill_advanced_stats(args.seasons)
    print("\nResult:", result)


if __name__ == "__main__":
    main()
