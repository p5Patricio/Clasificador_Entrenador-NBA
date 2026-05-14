from datetime import date

from app.services.elo import EloService


def test_elo_expected_probability_is_even_for_equal_ratings():
    assert EloService.expected_win_probability(1500, 1500) == 0.5


def test_elo_winner_rating_increases_and_loser_decreases():
    winner_after = EloService.updated_rating(1500, 1500, result=1.0, k_factor=20)
    loser_after = EloService.updated_rating(1500, 1500, result=0.0, k_factor=20)

    assert winner_after == 1510
    assert loser_after == 1490


def test_elo_build_team_timeline_processes_games_in_date_order():
    games = [
        {"game_id": "GAME-2", "game_date": date(2024, 1, 2), "result": "L", "opponent_team_id": 2},
        {"game_id": "GAME-1", "game_date": date(2024, 1, 1), "result": "W", "opponent_team_id": 2},
    ]

    timeline = EloService.build_team_timeline(team_id=1, season_id=10, games=games)

    assert [item.game_id for item in timeline] == ["GAME-1", "GAME-2"]
    assert timeline[0].rating_before == 1500
    assert timeline[0].rating_after == 1510
    assert timeline[1].rating_before == 1510
    assert timeline[1].rating_after < 1510
