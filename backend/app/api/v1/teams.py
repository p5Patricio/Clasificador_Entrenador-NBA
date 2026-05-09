from typing import List
from fastapi import APIRouter, Depends
from app.api.deps import get_team_service
from app.schemas import TeamResponse
from app.services.team import TeamService

router = APIRouter()


@router.get("/teams", response_model=List[TeamResponse])
def get_teams(service: TeamService = Depends(get_team_service)):
    return service.list_teams()
