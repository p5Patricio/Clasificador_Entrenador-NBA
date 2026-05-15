from fastapi import APIRouter, Depends

from app.api.deps import get_data_update_service
from app.schemas import DataUpdateRequest, DataUpdateResponse
from app.services.data_update import DataUpdateService

router = APIRouter()


@router.post("/data/update", response_model=DataUpdateResponse)
def data_update(
    request: DataUpdateRequest,
    service: DataUpdateService = Depends(get_data_update_service),
):
    summary = service.run_update(
        season=request.season,
        season_type=request.season_type,
        min_minutes=request.min_minutes,
        filepath=request.filepath,
    )
    return DataUpdateResponse(
        season=summary["season"],
        file=summary.get("file"),
        rows_processed=int(summary.get("rows_processed", 0)),
        players_inserted=int(summary.get("inserted", 0)),
        players_updated=int(summary.get("updated", 0)),
        players_failed=int(summary.get("failed", 0)),
        etl_status="completed" if int(summary.get("failed", 0)) == 0 else "completed_with_errors",
    )
