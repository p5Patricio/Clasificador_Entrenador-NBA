from fastapi import APIRouter, Depends, Query

from app.api.deps import get_historical_service
from app.schemas import (
    PlayerAdvancedStatsResponse,
    PlayerGameLogsResponse,
    PlayerSimilaritiesResponse,
    TeamEloResponse,
)
from app.services.historical import HistoricalService

router = APIRouter()


@router.get("/player/{player_name}/game-logs", response_model=PlayerGameLogsResponse)
def player_game_logs(
    player_name: str,
    season_id: int = Query(...),
    service: HistoricalService = Depends(get_historical_service),
):
    return service.get_player_game_logs(player_name, season_id)


@router.get("/player/{player_name}/advanced", response_model=PlayerAdvancedStatsResponse)
def player_advanced_stats(
    player_name: str,
    season_id: int = Query(...),
    service: HistoricalService = Depends(get_historical_service),
):
    return service.get_player_advanced_stats(player_name, season_id)


@router.get("/player/{player_name}/similar", response_model=PlayerSimilaritiesResponse)
def player_similarities(
    player_name: str,
    season_id: int = Query(...),
    service: HistoricalService = Depends(get_historical_service),
):
    return service.get_player_similarities(player_name, season_id)


@router.get("/teams/{team_abbreviation}/elo", response_model=TeamEloResponse)
def team_elo(
    team_abbreviation: str,
    season_id: int = Query(...),
    service: HistoricalService = Depends(get_historical_service),
):
    return service.get_team_elo(team_abbreviation, season_id)
