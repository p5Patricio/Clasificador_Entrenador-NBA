"""Create base tables for Phase 1

Revision ID: 001
Revises:
Create Date: 2026-04-29 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # season
    op.create_table(
        "season",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("season_label", sa.String(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("season_label"),
    )

    # team
    op.create_table(
        "team",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("abbreviation", sa.String(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=False),
        sa.Column("city", sa.String(), nullable=False),
        sa.Column("conference", sa.String(), nullable=False),
        sa.Column("division", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("abbreviation"),
    )
    op.create_index("ix_team_abbreviation", "team", ["abbreviation"], unique=False)

    # player
    op.create_table(
        "player",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("full_name", sa.String(), nullable=False),
        sa.Column("birthdate", sa.Date(), nullable=True),
        sa.Column("height_cm", sa.Integer(), nullable=True),
        sa.Column("weight_kg", sa.Integer(), nullable=True),
        sa.Column("position", sa.String(), nullable=True),
        sa.Column("headshot_url", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # player_season_stats
    op.create_table(
        "player_season_stats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=False),
        sa.Column("gp", sa.Integer(), nullable=True),
        sa.Column("gs", sa.Integer(), nullable=True),
        sa.Column("min", sa.Float(), nullable=True),
        sa.Column("fgm", sa.Integer(), nullable=True),
        sa.Column("fga", sa.Integer(), nullable=True),
        sa.Column("fg_pct", sa.Float(), nullable=True),
        sa.Column("fg3m", sa.Integer(), nullable=True),
        sa.Column("fg3a", sa.Integer(), nullable=True),
        sa.Column("fg3_pct", sa.Float(), nullable=True),
        sa.Column("ftm", sa.Integer(), nullable=True),
        sa.Column("fta", sa.Integer(), nullable=True),
        sa.Column("ft_pct", sa.Float(), nullable=True),
        sa.Column("oreb", sa.Integer(), nullable=True),
        sa.Column("dreb", sa.Integer(), nullable=True),
        sa.Column("reb", sa.Integer(), nullable=True),
        sa.Column("ast", sa.Integer(), nullable=True),
        sa.Column("stl", sa.Integer(), nullable=True),
        sa.Column("blk", sa.Integer(), nullable=True),
        sa.Column("tov", sa.Integer(), nullable=True),
        sa.Column("pf", sa.Integer(), nullable=True),
        sa.Column("pts", sa.Integer(), nullable=True),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["player_id"], ["player.id"]),
        sa.ForeignKeyConstraint(["team_id"], ["team.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["season.id"]),
        sa.UniqueConstraint("player_id", "season_id", "team_id"),
    )
    op.create_index(
        "ix_pss_season_cluster", "player_season_stats", ["season_id", "cluster_id"], unique=False
    )
    op.create_index(
        "ix_player_season_stats_player_id", "player_season_stats", ["player_id"], unique=False
    )

    # team_season_stats
    op.create_table(
        "team_season_stats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=False),
        sa.Column("wins", sa.Integer(), nullable=True),
        sa.Column("losses", sa.Integer(), nullable=True),
        sa.Column("win_pct", sa.Float(), nullable=True),
        sa.Column("pts_avg", sa.Float(), nullable=True),
        sa.Column("reb_avg", sa.Float(), nullable=True),
        sa.Column("ast_avg", sa.Float(), nullable=True),
        sa.Column("extra_stats", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["team_id"], ["team.id"]),
        sa.ForeignKeyConstraint(["season_id"], ["season.id"]),
        sa.UniqueConstraint("team_id", "season_id"),
    )


def downgrade() -> None:
    op.drop_table("team_season_stats")
    op.drop_index("ix_pss_season_cluster", table_name="player_season_stats")
    op.drop_index("ix_player_season_stats_player_id", table_name="player_season_stats")
    op.drop_table("player_season_stats")
    op.drop_table("player")
    op.drop_index("ix_team_abbreviation", table_name="team")
    op.drop_table("team")
    op.drop_table("season")
