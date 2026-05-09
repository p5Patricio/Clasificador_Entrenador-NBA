from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class TeamResponse(BaseModel):
    id: int
    abbreviation: str
    full_name: str
    city: str
    conference: str
    division: str
    players: int


class PlayerSummary(BaseModel):
    id: int
    full_name: str
    team_abbreviation: Optional[str]


class StatDelta(BaseModel):
    stat: str
    player_value: float
    cluster_avg: float
    diff: float
    percent_diff: float


class PlayerAnalysisResponse(BaseModel):
    player_name: str
    team: Optional[str]
    cluster_id: Optional[int]
    comparison: List[StatDelta]


class PlayerRadarsResponse(BaseModel):
    player_name: str
    seasons: List[Dict[str, Any]]


class ClusterInitResponse(BaseModel):
    season_id: int
    k: int
    players: int
    clusters: int
    roles: Dict[int, str]


class PlayerShot(BaseModel):
    x: float
    y: float
    made: bool
    action_type: Optional[str] = None
    shot_zone_basic: Optional[str] = None
    shot_distance: Optional[float] = None


class PlayerShotsResponse(BaseModel):
    season: str
    season_type: str
    attempts: int
    makes: int
    shots: List[PlayerShot]


class PlayerProfileResponse(BaseModel):
    player_id: int
    full_name: str
    team_id: Optional[int]
    team_abbreviation: Optional[str]
    team_name: Optional[str]
    height: Optional[str]
    weight: Optional[str]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    birthdate: Optional[str]
    age: Optional[float]
    headshot_url: Optional[str]


class HealthResponse(BaseModel):
    status: str
    db_connected: bool


class DataUpdateResponse(BaseModel):
    season: str
    players_upserted: int
    etl_status: str
