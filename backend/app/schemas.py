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


class PlayerGameLogItem(BaseModel):
    game_id: str
    game_date: Optional[str]
    matchup: Optional[str]
    wl: Optional[str]
    min: int
    pts: int
    reb: int
    ast: int
    stl: int
    blk: int
    plus_minus: int


class PlayerGameLogsResponse(BaseModel):
    player_name: str
    season_id: int
    games: List[PlayerGameLogItem]


class HistoricalPlayerShotsResponse(BaseModel):
    player_name: str
    season_id: int
    attempts: int
    makes: int
    shots: List[PlayerShot]


class PlayerAdvancedStatsResponse(BaseModel):
    player_name: str
    season_id: int
    per: Optional[float] = None
    ts_pct: Optional[float] = None
    ftr: Optional[float] = None
    orb_pct: Optional[float] = None
    drb_pct: Optional[float] = None
    trb_pct: Optional[float] = None
    ast_pct: Optional[float] = None
    stl_pct: Optional[float] = None
    blk_pct: Optional[float] = None
    tov_pct: Optional[float] = None
    usg_pct: Optional[float] = None
    ows: Optional[float] = None
    dws: Optional[float] = None
    ws: Optional[float] = None
    ws_per_48: Optional[float] = None
    obpm: Optional[float] = None
    dbpm: Optional[float] = None
    bpm: Optional[float] = None
    vorp: Optional[float] = None


class SimilarPlayerItem(BaseModel):
    player_id: int
    player_name: str
    similarity_score: float


class PlayerSimilaritiesResponse(BaseModel):
    player_name: str
    season_id: int
    players: List[SimilarPlayerItem]


class TeamEloItem(BaseModel):
    game_id: Optional[str]
    game_date: Optional[str]
    opponent_team_id: Optional[int]
    rating_before: float
    rating_after: float
    result: Optional[str]
    win_probability: Optional[float]


class TeamEloResponse(BaseModel):
    team_abbreviation: str
    season_id: int
    timeline: List[TeamEloItem]


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
