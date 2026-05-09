from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_stats_repo
from app.schemas import DataUpdateResponse
from app.repositories.stats import StatsRepository

router = APIRouter()


@router.post("/data/update", response_model=DataUpdateResponse)
def data_update(
    season: str,
    season_type: str = "Regular Season",
    min_minutes: int = 100,
    stats_repo: StatsRepository = Depends(get_stats_repo),
):
    # Placeholder: ETL pipeline will be wired here
    # For now, return a stub response indicating ETL is not yet implemented
    return DataUpdateResponse(
        season=season,
        players_upserted=0,
        etl_status="ETL pipeline not yet implemented",
    )
