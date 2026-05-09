import os
import pandas as pd
from typing import Optional, Dict, Any, List
from sqlmodel import Session, select
from app.models import Player, Team, Season, PlayerSeasonStats


BATCH_SIZE = 100


class ETLPipeline:
    def __init__(self, session: Session):
        self.session = session

    def run(self, filepath: str, season_label: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Excel file not found: {filepath}")

        df = pd.read_excel(filepath)
        required_cols = {"PLAYER_NAME", "TEAM_ABBREVIATION"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Use SEASON_ID from file if present, otherwise use passed season_label
        if "SEASON_ID" in df.columns and df["SEASON_ID"].notna().any():
            file_season = str(df["SEASON_ID"].iloc[0])
            if file_season and file_season != "nan":
                season_label = file_season

        season = self._get_or_create_season(season_label)

        summary = {"inserted": 0, "updated": 0, "failed": 0}
        batch_players: Dict[int, Player] = {}
        batch_teams: Dict[str, Team] = {}

        for idx, row in df.iterrows():
            try:
                player = self._upsert_player(row, batch_players)
                team = self._upsert_team(row, batch_teams)
                stat = self._build_stats(row, player.id, team.id, season.id)

                existing = self.session.exec(
                    select(PlayerSeasonStats).where(
                        PlayerSeasonStats.player_id == stat.player_id,
                        PlayerSeasonStats.season_id == stat.season_id,
                        PlayerSeasonStats.team_id == stat.team_id,
                    )
                ).first()

                if existing:
                    for key, value in stat.model_dump(exclude={"id"}).items():
                        setattr(existing, key, value)
                    self.session.add(existing)
                    summary["updated"] += 1
                else:
                    self.session.add(stat)
                    summary["inserted"] += 1

                # Commit every BATCH_SIZE rows to avoid huge transactions
                if (idx + 1) % BATCH_SIZE == 0:
                    self.session.commit()
                    batch_players.clear()
                    batch_teams.clear()

            except Exception as e:
                summary["failed"] += 1
                print(f"Error processing row {idx}: {e}")
                self.session.rollback()
                continue

        # Final commit for remaining rows
        self.session.commit()

        return {
            "season": season_label,
            "file": filepath,
            "rows_processed": len(df),
            **summary,
        }

    def _get_or_create_season(self, season_label: str) -> Season:
        stmt = select(Season).where(Season.season_label == season_label)
        result = self.session.exec(stmt).first()
        if result:
            return result
        season = Season(season_label=season_label)
        self.session.add(season)
        self.session.commit()
        self.session.refresh(season)
        return season

    def _upsert_player(self, row: pd.Series, cache: Dict[int, Player]) -> Player:
        player_id = int(row.get("PLAYER_ID", 0)) if pd.notna(row.get("PLAYER_ID")) else 0
        name = str(row.get("PLAYER_NAME", ""))
        if not player_id:
            player_id = abs(hash(name)) % (10**9)

        if player_id in cache:
            return cache[player_id]

        existing = self.session.get(Player, player_id)
        if existing:
            cache[player_id] = existing
            return existing

        player = Player(id=player_id, full_name=name)
        self.session.add(player)
        self.session.flush()
        cache[player_id] = player
        return player

    def _upsert_team(self, row: pd.Series, cache: Dict[str, Team]) -> Team:
        abbr = str(row.get("TEAM_ABBREVIATION", ""))
        if not abbr or abbr == "nan":
            abbr = "TOT"

        if abbr in cache:
            return cache[abbr]

        existing = self.session.exec(
            select(Team).where(Team.abbreviation == abbr)
        ).first()
        if existing:
            cache[abbr] = existing
            return existing

        team = Team(
            id=abs(hash(abbr)) % (10**9),
            abbreviation=abbr,
            full_name=abbr,
            city="",
            conference="",
            division="",
        )
        self.session.add(team)
        self.session.flush()
        cache[abbr] = team
        return team

    def _build_stats(self, row: pd.Series, player_id: int, team_id: int, season_id: int) -> PlayerSeasonStats:
        def get_val(col: str, default=0):
            val = row.get(col, default)
            if pd.isna(val):
                return default
            return val

        return PlayerSeasonStats(
            player_id=player_id,
            team_id=team_id,
            season_id=season_id,
            gp=int(get_val("GP", 0)),
            gs=int(get_val("GS", 0)),
            min=float(get_val("MIN", 0)),
            fgm=float(get_val("FGM", 0)),
            fga=float(get_val("FGA", 0)),
            fg_pct=float(get_val("FG_PCT", 0)),
            fg3m=float(get_val("FG3M", 0)),
            fg3a=float(get_val("FG3A", 0)),
            fg3_pct=float(get_val("FG3_PCT", 0)),
            ftm=float(get_val("FTM", 0)),
            fta=float(get_val("FTA", 0)),
            ft_pct=float(get_val("FT_PCT", 0)),
            oreb=float(get_val("OREB", 0)),
            dreb=float(get_val("DREB", 0)),
            reb=float(get_val("REB", 0)),
            ast=float(get_val("AST", 0)),
            stl=float(get_val("STL", 0)),
            blk=float(get_val("BLK", 0)),
            tov=float(get_val("TOV", 0)),
            pf=float(get_val("PF", 0)),
            pts=float(get_val("PTS", 0)),
        )
