from typing import Optional, Dict, Any, List
from fastapi import HTTPException
from app.models import Player
from app.repositories.player import PlayerRepository
from app.repositories.stats import StatsRepository


class PlayerService:
    def __init__(self, player_repo: PlayerRepository, stats_repo: StatsRepository):
        self.player_repo = player_repo
        self.stats_repo = stats_repo

    def get_analysis(self, player_name: str, season_id: Optional[int] = None) -> Dict[str, Any]:
        player = self.player_repo.get_by_name(player_name)
        if not player:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

        stats_list = self.stats_repo.find_by_season(season_id) if season_id else player.season_stats
        if not stats_list:
            raise HTTPException(status_code=404, detail=f"No stats found for player in season {season_id}")

        player_stats = stats_list[0] if isinstance(stats_list, list) else stats_list

        all_season_stats = self.stats_repo.find_by_season(player_stats.season_id)
        cluster_stats = [s for s in all_season_stats if s.cluster_id == player_stats.cluster_id and s.player_id != player.id]

        comparison = []
        stat_fields = [
            "gp", "gs", "min", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
            "ftm", "fta", "ft_pct", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts",
        ]

        for field in stat_fields:
            p_val = getattr(player_stats, field, 0)
            c_vals = [getattr(s, field, 0) for s in cluster_stats if getattr(s, field, 0) is not None]
            c_avg = sum(c_vals) / len(c_vals) if c_vals else 0
            diff = p_val - c_avg
            percent = (diff / c_avg * 100) if c_avg else 0
            comparison.append({
                "stat": field,
                "player_value": p_val,
                "cluster_avg": c_avg,
                "diff": diff,
                "percent_diff": percent,
            })

        return {
            "player_name": player.full_name,
            "team": player_stats.team.abbreviation if player_stats.team else None,
            "cluster_id": player_stats.cluster_id,
            "comparison": comparison,
        }

    def get_radars(self, player_name: str) -> Dict[str, Any]:
        player = self.player_repo.get_by_name(player_name)
        if not player:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

        stats = player.season_stats
        return {
            "player_name": player.full_name,
            "seasons": [
                {
                    "season_id": s.season_id,
                    "season_label": s.season.season_label if s.season else None,
                    "stats": {
                        "pts": s.pts, "ast": s.ast, "reb": s.reb,
                        "stl": s.stl, "blk": s.blk, "tov": s.tov,
                        "fg_pct": s.fg_pct, "fg3_pct": s.fg3_pct, "ft_pct": s.ft_pct,
                    },
                }
                for s in stats
            ],
        }
