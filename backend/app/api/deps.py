from fastapi import Depends
from sqlmodel import Session
from app.deps import get_db_session
from app.repositories.player import PlayerRepository
from app.repositories.team import TeamRepository
from app.repositories.stats import StatsRepository
from app.repositories.historical import HistoricalRepository
from app.services.player import PlayerService
from app.services.team import TeamService
from app.services.clustering import ClusteringService
from app.services.report import ReportService
from app.services.historical import HistoricalService


def get_report_service(session: Session = Depends(get_db_session)) -> ReportService:
    return ReportService(session)


def get_player_repo(session: Session = Depends(get_db_session)) -> PlayerRepository:
    return PlayerRepository(session)


def get_team_repo(session: Session = Depends(get_db_session)) -> TeamRepository:
    return TeamRepository(session)


def get_stats_repo(session: Session = Depends(get_db_session)) -> StatsRepository:
    return StatsRepository(session)


def get_historical_repo(session: Session = Depends(get_db_session)) -> HistoricalRepository:
    return HistoricalRepository(session)


def get_player_service(
    player_repo: PlayerRepository = Depends(get_player_repo),
    stats_repo: StatsRepository = Depends(get_stats_repo),
) -> PlayerService:
    return PlayerService(player_repo, stats_repo)


def get_team_service(
    team_repo: TeamRepository = Depends(get_team_repo),
) -> TeamService:
    return TeamService(team_repo)


def get_clustering_service(
    stats_repo: StatsRepository = Depends(get_stats_repo),
) -> ClusteringService:
    return ClusteringService(stats_repo)


def get_historical_service(
    player_repo: PlayerRepository = Depends(get_player_repo),
    team_repo: TeamRepository = Depends(get_team_repo),
    historical_repo: HistoricalRepository = Depends(get_historical_repo),
) -> HistoricalService:
    return HistoricalService(player_repo, team_repo, historical_repo)
