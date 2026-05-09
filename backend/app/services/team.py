from typing import List, Dict, Any
from app.repositories.team import TeamRepository


class TeamService:
    def __init__(self, team_repo: TeamRepository):
        self.team_repo = team_repo

    def list_teams(self) -> List[Dict[str, Any]]:
        teams = self.team_repo.list()
        result = []
        for team in teams:
            player_count = len(team.player_stats) if team.player_stats else 0
            result.append({
                "id": team.id,
                "abbreviation": team.abbreviation,
                "full_name": team.full_name,
                "city": team.city,
                "conference": team.conference,
                "division": team.division,
                "players": player_count,
            })
        return result
