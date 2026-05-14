from fastapi import HTTPException

from app.repositories.historical import HistoricalRepository
from app.repositories.player import PlayerRepository
from app.repositories.team import TeamRepository
from app.schemas import (
    HistoricalPlayerShotsResponse,
    PlayerAdvancedStatsResponse,
    PlayerGameLogItem,
    PlayerGameLogsResponse,
    PlayerShot,
    PlayerSimilaritiesResponse,
    SimilarPlayerItem,
    TeamEloItem,
    TeamEloResponse,
)


class HistoricalService:
    def __init__(
        self,
        player_repo: PlayerRepository,
        team_repo: TeamRepository,
        historical_repo: HistoricalRepository,
    ):
        self.player_repo = player_repo
        self.team_repo = team_repo
        self.historical_repo = historical_repo

    def _get_player_or_404(self, player_name: str):
        player = self.player_repo.get_by_name(player_name)
        if not player:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
        return player

    def _get_team_or_404(self, team_abbreviation: str):
        team = self.team_repo.get_by_abbreviation(team_abbreviation)
        if not team:
            raise HTTPException(status_code=404, detail=f"Team '{team_abbreviation}' not found")
        return team

    def get_player_game_logs(self, player_name: str, season_id: int) -> PlayerGameLogsResponse:
        player = self._get_player_or_404(player_name)
        logs = self.historical_repo.list_player_game_logs(player.id, season_id)
        return PlayerGameLogsResponse(
            player_name=player.full_name,
            season_id=season_id,
            games=[
                PlayerGameLogItem(
                    game_id=log.game_id,
                    game_date=log.game_date.isoformat() if log.game_date else None,
                    matchup=log.matchup,
                    wl=log.wl,
                    min=log.min,
                    pts=log.pts,
                    reb=log.reb,
                    ast=log.ast,
                    stl=log.stl,
                    blk=log.blk,
                    plus_minus=log.plus_minus,
                )
                for log in logs
            ],
        )

    def get_player_shots(self, player_name: str, season_id: int) -> HistoricalPlayerShotsResponse:
        player = self._get_player_or_404(player_name)
        shots = self.historical_repo.list_player_shots(player.id, season_id)
        return HistoricalPlayerShotsResponse(
            player_name=player.full_name,
            season_id=season_id,
            attempts=len(shots),
            makes=sum(1 for shot in shots if shot.shot_made_flag),
            shots=[
                PlayerShot(
                    x=shot.loc_x,
                    y=shot.loc_y,
                    made=shot.shot_made_flag,
                    action_type=shot.action_type,
                    shot_zone_basic=shot.shot_zone_basic,
                    shot_distance=shot.shot_distance,
                )
                for shot in shots
            ],
        )

    def get_player_advanced_stats(self, player_name: str, season_id: int) -> PlayerAdvancedStatsResponse:
        player = self._get_player_or_404(player_name)
        stats = self.historical_repo.get_player_advanced_stats(player.id, season_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Advanced stats not found")
        return PlayerAdvancedStatsResponse(
            player_name=player.full_name,
            season_id=season_id,
            per=stats.per,
            ts_pct=stats.ts_pct,
            ftr=stats.ftr,
            orb_pct=stats.orb_pct,
            drb_pct=stats.drb_pct,
            trb_pct=stats.trb_pct,
            ast_pct=stats.ast_pct,
            stl_pct=stats.stl_pct,
            blk_pct=stats.blk_pct,
            tov_pct=stats.tov_pct,
            usg_pct=stats.usg_pct,
            ows=stats.ows,
            dws=stats.dws,
            ws=stats.ws,
            ws_per_48=stats.ws_per_48,
            obpm=stats.obpm,
            dbpm=stats.dbpm,
            bpm=stats.bpm,
            vorp=stats.vorp,
        )

    def get_player_similarities(self, player_name: str, season_id: int) -> PlayerSimilaritiesResponse:
        player = self._get_player_or_404(player_name)
        similarities = self.historical_repo.list_player_similarities(player.id, season_id)
        return PlayerSimilaritiesResponse(
            player_name=player.full_name,
            season_id=season_id,
            players=[
                SimilarPlayerItem(
                    player_id=item.similar_player_id,
                    player_name=(
                        self.player_repo.get(item.similar_player_id).full_name
                        if self.player_repo.get(item.similar_player_id)
                        else str(item.similar_player_id)
                    ),
                    similarity_score=item.similarity_score,
                )
                for item in similarities
            ],
        )

    def get_team_elo(self, team_abbreviation: str, season_id: int) -> TeamEloResponse:
        team = self._get_team_or_404(team_abbreviation)
        timeline = self.historical_repo.list_team_elo(team.id, season_id)
        return TeamEloResponse(
            team_abbreviation=team.abbreviation,
            season_id=season_id,
            timeline=[
                TeamEloItem(
                    game_id=item.game_id,
                    game_date=item.game_date.isoformat() if item.game_date else None,
                    opponent_team_id=item.opponent_team_id,
                    rating_before=item.rating_before,
                    rating_after=item.rating_after,
                    result=item.result,
                    win_probability=item.win_probability,
                )
                for item in timeline
            ],
        )
