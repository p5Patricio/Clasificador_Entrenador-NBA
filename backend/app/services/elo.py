from typing import Iterable, Mapping

from app.models import TeamEloRating


class EloService:
    @staticmethod
    def expected_win_probability(rating: float, opponent_rating: float) -> float:
        return round(1 / (1 + 10 ** ((opponent_rating - rating) / 400)), 4)

    @staticmethod
    def updated_rating(
        rating: float,
        opponent_rating: float,
        result: float,
        k_factor: float = 20.0,
    ) -> float:
        expected = EloService.expected_win_probability(rating, opponent_rating)
        return round(rating + k_factor * (result - expected), 2)

    @staticmethod
    def build_team_timeline(
        team_id: int,
        season_id: int,
        games: Iterable[Mapping],
        initial_rating: float = 1500.0,
        opponent_initial_rating: float = 1500.0,
        k_factor: float = 20.0,
    ) -> list[TeamEloRating]:
        current_rating = initial_rating
        timeline: list[TeamEloRating] = []

        for game in sorted(games, key=lambda item: item.get("game_date")):
            result_label = game.get("result")
            result = 1.0 if result_label == "W" else 0.0
            win_probability = EloService.expected_win_probability(current_rating, opponent_initial_rating)
            rating_after = EloService.updated_rating(
                current_rating,
                opponent_initial_rating,
                result,
                k_factor,
            )
            timeline.append(
                TeamEloRating(
                    team_id=team_id,
                    season_id=season_id,
                    game_id=game.get("game_id"),
                    game_date=game.get("game_date"),
                    opponent_team_id=game.get("opponent_team_id"),
                    rating_before=current_rating,
                    rating_after=rating_after,
                    k_factor=k_factor,
                    result=result_label,
                    win_probability=win_probability,
                )
            )
            current_rating = rating_after

        return timeline
