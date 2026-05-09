from typing import Optional
from sqlmodel import Session, select
from app.models import Team
from app.repositories.base import BaseRepository


class TeamRepository(BaseRepository[Team]):
    def __init__(self, session: Session):
        super().__init__(session, Team)

    def get_by_abbreviation(self, abbr: str) -> Optional[Team]:
        stmt = select(Team).where(Team.abbreviation == abbr)
        return self.session.exec(stmt).first()
