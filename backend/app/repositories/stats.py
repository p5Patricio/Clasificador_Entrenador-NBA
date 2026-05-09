from typing import List, Optional, Dict
from sqlmodel import Session, select
from app.models import PlayerSeasonStats, TeamSeasonStats, Team


class StatsRepository:
    def __init__(self, session: Session):
        self.session = session

    def upsert_player_stats(self, stats: PlayerSeasonStats) -> PlayerSeasonStats:
        """Insert or update player season stats."""
        # Check if exists
        stmt = select(PlayerSeasonStats).where(
            PlayerSeasonStats.player_id == stats.player_id,
            PlayerSeasonStats.season_id == stats.season_id,
            PlayerSeasonStats.team_id == stats.team_id,
        )
        existing = self.session.exec(stmt).first()
        if existing:
            # Update fields
            for key, value in stats.model_dump(exclude={"id"}).items():
                setattr(existing, key, value)
            self.session.add(existing)
            self.session.commit()
            self.session.refresh(existing)
            return existing
        else:
            self.session.add(stats)
            self.session.commit()
            self.session.refresh(stats)
            return stats

    def find_by_season(
        self, season_id: int, team_abbreviation: Optional[str] = None
    ) -> List[PlayerSeasonStats]:
        stmt = select(PlayerSeasonStats).where(PlayerSeasonStats.season_id == season_id)
        if team_abbreviation:
            stmt = stmt.join(Team).where(Team.abbreviation == team_abbreviation)
        return self.session.exec(stmt).all()

    def update_clusters(self, season_id: int, assignments: Dict[int, int]) -> int:
        """Update cluster_id for player_season_stats. assignments: {stats_id: cluster_id}"""
        count = 0
        for stats_id, cluster_id in assignments.items():
            stats = self.session.get(PlayerSeasonStats, stats_id)
            if stats:
                stats.cluster_id = cluster_id
                self.session.add(stats)
                count += 1
        self.session.commit()
        return count
