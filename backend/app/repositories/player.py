from typing import Optional, List
from sqlmodel import Session, select
from app.models import Player, PlayerSeasonStats
from app.repositories.base import BaseRepository


class PlayerRepository(BaseRepository[Player]):
    def __init__(self, session: Session):
        super().__init__(session, Player)

    def get_by_name(self, name: str) -> Optional[Player]:
        stmt = select(Player).where(Player.full_name == name)
        return self.session.exec(stmt).first()

    def list_with_stats(self, season_id: Optional[int] = None) -> List[Player]:
        stmt = select(Player)
        if season_id:
            stmt = stmt.join(PlayerSeasonStats).where(PlayerSeasonStats.season_id == season_id)
        return self.session.exec(stmt).all()
