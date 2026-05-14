import math
from typing import Iterable

from app.models import PlayerSeasonStats, PlayerSimilarity


SIMILARITY_FIELDS = (
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
)


class SimilarityService:
    @staticmethod
    def _vector(stats: PlayerSeasonStats) -> list[float]:
        return [float(getattr(stats, field, 0.0) or 0.0) for field in SIMILARITY_FIELDS]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def rank_similar_players(
        target: PlayerSeasonStats,
        candidates: Iterable[PlayerSeasonStats],
        top_n: int = 10,
    ) -> list[PlayerSimilarity]:
        target_vector = SimilarityService._vector(target)
        similarities = []

        for candidate in candidates:
            if candidate.player_id == target.player_id:
                continue
            score = SimilarityService._cosine_similarity(
                target_vector,
                SimilarityService._vector(candidate),
            )
            similarities.append(
                PlayerSimilarity(
                    player_id=target.player_id,
                    season_id=target.season_id,
                    similar_player_id=candidate.player_id,
                    similarity_score=round(score, 4),
                )
            )

        return sorted(similarities, key=lambda item: item.similarity_score, reverse=True)[:top_n]
