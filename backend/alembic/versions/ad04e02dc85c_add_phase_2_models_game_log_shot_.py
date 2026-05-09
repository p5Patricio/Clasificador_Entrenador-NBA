"""Add phase 2 models: game_log, shot, advanced_stats, elo, similarity

Revision ID: ad04e02dc85c
Revises: 001
Create Date: 2026-05-09 14:36:28.696387

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ad04e02dc85c'
down_revision: Union[str, Sequence[str], None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create new Phase 2 tables only — do NOT alter existing tables
    op.create_table('player_advanced_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        sa.Column('season_id', sa.Integer(), nullable=False),
        sa.Column('per', sa.Float(), nullable=True),
        sa.Column('ts_pct', sa.Float(), nullable=True),
        sa.Column('ftr', sa.Float(), nullable=True),
        sa.Column('orb_pct', sa.Float(), nullable=True),
        sa.Column('drb_pct', sa.Float(), nullable=True),
        sa.Column('trb_pct', sa.Float(), nullable=True),
        sa.Column('ast_pct', sa.Float(), nullable=True),
        sa.Column('stl_pct', sa.Float(), nullable=True),
        sa.Column('blk_pct', sa.Float(), nullable=True),
        sa.Column('tov_pct', sa.Float(), nullable=True),
        sa.Column('usg_pct', sa.Float(), nullable=True),
        sa.Column('ows', sa.Float(), nullable=True),
        sa.Column('dws', sa.Float(), nullable=True),
        sa.Column('ws', sa.Float(), nullable=True),
        sa.Column('ws_per_48', sa.Float(), nullable=True),
        sa.Column('obpm', sa.Float(), nullable=True),
        sa.Column('dbpm', sa.Float(), nullable=True),
        sa.Column('bpm', sa.Float(), nullable=True),
        sa.Column('vorp', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['player_id'], ['player.id'], ),
        sa.ForeignKeyConstraint(['season_id'], ['season.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('player_id', 'season_id', name='uq_player_season_advanced')
    )

    op.create_table('player_game_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('season_id', sa.Integer(), nullable=False),
        sa.Column('game_id', sa.String(), nullable=False),
        sa.Column('game_date', sa.Date(), nullable=True),
        sa.Column('matchup', sa.String(), nullable=True),
        sa.Column('wl', sa.String(), nullable=True),
        sa.Column('min', sa.Integer(), nullable=False),
        sa.Column('fgm', sa.Integer(), nullable=False),
        sa.Column('fga', sa.Integer(), nullable=False),
        sa.Column('fg_pct', sa.Float(), nullable=False),
        sa.Column('fg3m', sa.Integer(), nullable=False),
        sa.Column('fg3a', sa.Integer(), nullable=False),
        sa.Column('fg3_pct', sa.Float(), nullable=False),
        sa.Column('ftm', sa.Integer(), nullable=False),
        sa.Column('fta', sa.Integer(), nullable=False),
        sa.Column('ft_pct', sa.Float(), nullable=False),
        sa.Column('oreb', sa.Integer(), nullable=False),
        sa.Column('dreb', sa.Integer(), nullable=False),
        sa.Column('reb', sa.Integer(), nullable=False),
        sa.Column('ast', sa.Integer(), nullable=False),
        sa.Column('stl', sa.Integer(), nullable=False),
        sa.Column('blk', sa.Integer(), nullable=False),
        sa.Column('tov', sa.Integer(), nullable=False),
        sa.Column('pf', sa.Integer(), nullable=False),
        sa.Column('pts', sa.Integer(), nullable=False),
        sa.Column('plus_minus', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['player_id'], ['player.id'], ),
        sa.ForeignKeyConstraint(['season_id'], ['season.id'], ),
        sa.ForeignKeyConstraint(['team_id'], ['team.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('player_id', 'game_id', name='uq_player_game')
    )
    op.create_index('ix_game_season', 'player_game_log', ['season_id', 'game_date'], unique=False)
    op.create_index(op.f('ix_player_game_log_game_id'), 'player_game_log', ['game_id'], unique=False)

    op.create_table('player_shot',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        sa.Column('season_id', sa.Integer(), nullable=False),
        sa.Column('game_id', sa.String(), nullable=True),
        sa.Column('game_event_id', sa.Integer(), nullable=True),
        sa.Column('period', sa.Integer(), nullable=False),
        sa.Column('minutes_remaining', sa.Integer(), nullable=False),
        sa.Column('seconds_remaining', sa.Integer(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=True),
        sa.Column('action_type', sa.String(), nullable=True),
        sa.Column('shot_type', sa.String(), nullable=True),
        sa.Column('shot_zone_basic', sa.String(), nullable=True),
        sa.Column('shot_zone_area', sa.String(), nullable=True),
        sa.Column('shot_zone_range', sa.String(), nullable=True),
        sa.Column('shot_distance', sa.Float(), nullable=False),
        sa.Column('loc_x', sa.Float(), nullable=False),
        sa.Column('loc_y', sa.Float(), nullable=False),
        sa.Column('shot_attempted_flag', sa.Boolean(), nullable=False),
        sa.Column('shot_made_flag', sa.Boolean(), nullable=False),
        sa.Column('game_date', sa.Date(), nullable=True),
        sa.ForeignKeyConstraint(['player_id'], ['player.id'], ),
        sa.ForeignKeyConstraint(['season_id'], ['season.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_shot_game', 'player_shot', ['game_id'], unique=False)
    op.create_index('ix_shot_player_season', 'player_shot', ['player_id', 'season_id'], unique=False)

    op.create_table('player_similarity',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('player_id', sa.Integer(), nullable=False),
        sa.Column('season_id', sa.Integer(), nullable=False),
        sa.Column('similar_player_id', sa.Integer(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['player_id'], ['player.id'], ),
        sa.ForeignKeyConstraint(['season_id'], ['season.id'], ),
        sa.ForeignKeyConstraint(['similar_player_id'], ['player.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('player_id', 'season_id', 'similar_player_id', name='uq_similarity')
    )
    op.create_index('ix_similarity_score', 'player_similarity', ['season_id', 'similarity_score'], unique=False)

    op.create_table('team_elo_rating',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('team_id', sa.Integer(), nullable=False),
        sa.Column('season_id', sa.Integer(), nullable=False),
        sa.Column('game_id', sa.String(), nullable=True),
        sa.Column('game_date', sa.Date(), nullable=True),
        sa.Column('opponent_team_id', sa.Integer(), nullable=True),
        sa.Column('rating_before', sa.Float(), nullable=False),
        sa.Column('rating_after', sa.Float(), nullable=False),
        sa.Column('k_factor', sa.Float(), nullable=False),
        sa.Column('result', sa.String(), nullable=True),
        sa.Column('win_probability', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['season_id'], ['season.id'], ),
        sa.ForeignKeyConstraint(['team_id'], ['team.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('team_id', 'season_id', 'game_id', name='uq_team_season_game_elo')
    )
    op.create_index('ix_elo_season', 'team_elo_rating', ['season_id', 'rating_after'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_elo_season', table_name='team_elo_rating')
    op.drop_table('team_elo_rating')
    op.drop_index('ix_similarity_score', table_name='player_similarity')
    op.drop_table('player_similarity')
    op.drop_index('ix_shot_player_season', table_name='player_shot')
    op.drop_index('ix_shot_game', table_name='player_shot')
    op.drop_table('player_shot')
    op.drop_index(op.f('ix_player_game_log_game_id'), table_name='player_game_log')
    op.drop_index('ix_game_season', table_name='player_game_log')
    op.drop_table('player_game_log')
    op.drop_table('player_advanced_stats')
