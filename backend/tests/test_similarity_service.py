from app.models import PlayerSeasonStats
from app.services.similarity import SimilarityService


def test_similarity_scores_rank_closest_player_first():
    target = PlayerSeasonStats(
        id=1,
        player_id=1,
        team_id=1,
        season_id=10,
        pts=25,
        reb=8,
        ast=7,
        stl=1.2,
        blk=0.8,
        fg_pct=0.50,
        fg3_pct=0.38,
        ft_pct=0.82,
    )
    close = PlayerSeasonStats(
        id=2,
        player_id=2,
        team_id=1,
        season_id=10,
        pts=24,
        reb=8,
        ast=7,
        stl=1.1,
        blk=0.7,
        fg_pct=0.49,
        fg3_pct=0.37,
        ft_pct=0.81,
    )
    far = PlayerSeasonStats(
        id=3,
        player_id=3,
        team_id=1,
        season_id=10,
        pts=8,
        reb=2,
        ast=1,
        stl=0.3,
        blk=0.1,
        fg_pct=0.39,
        fg3_pct=0.25,
        ft_pct=0.61,
    )

    scores = SimilarityService.rank_similar_players(target, [far, close])

    assert scores[0].similar_player_id == 2
    assert scores[0].similarity_score > scores[1].similarity_score


def test_similarity_service_limits_top_n_results():
    target = PlayerSeasonStats(id=1, player_id=1, team_id=1, season_id=10, pts=10)
    candidates = [
        PlayerSeasonStats(id=2, player_id=2, team_id=1, season_id=10, pts=11),
        PlayerSeasonStats(id=3, player_id=3, team_id=1, season_id=10, pts=12),
    ]

    scores = SimilarityService.rank_similar_players(target, candidates, top_n=1)

    assert len(scores) == 1
