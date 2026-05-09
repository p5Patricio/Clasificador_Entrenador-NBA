from datetime import date
from sqlmodel import Session
from app.models import Player, Team, Season, PlayerSeasonStats
from app.repositories.player import PlayerRepository
from app.repositories.team import TeamRepository
from app.repositories.stats import StatsRepository


def test_player_repo_create_and_get(db_session: Session):
    repo = PlayerRepository(db_session)
    player = Player(id=1, full_name="Test Player")
    created = repo.add(player)
    assert created.id == 1
    assert created.full_name == "Test Player"

    fetched = repo.get(1)
    assert fetched is not None
    assert fetched.full_name == "Test Player"


def test_player_repo_get_by_name(db_session: Session):
    repo = PlayerRepository(db_session)
    repo.add(Player(id=2, full_name="LeBron James"))
    found = repo.get_by_name("LeBron James")
    assert found is not None
    assert found.id == 2

    not_found = repo.get_by_name("Unknown")
    assert not_found is None


def test_team_repo_create_and_get_by_abbr(db_session: Session):
    repo = TeamRepository(db_session)
    team = Team(id=1, abbreviation="LAL", full_name="Lakers", city="Los Angeles", conference="West", division="Pacific")
    repo.add(team)

    found = repo.get_by_abbreviation("LAL")
    assert found is not None
    assert found.full_name == "Lakers"


def test_stats_repo_upsert_and_find(db_session: Session):
    # Setup dependencies
    season = Season(id=99, season_label="2023-24")
    team = Team(id=99, abbreviation="TST", full_name="Testers", city="Test City", conference="East", division="Atlantic")
    player = Player(id=99, full_name="Test Player")
    db_session.add_all([season, team, player])
    db_session.commit()

    repo = StatsRepository(db_session)
    stats = PlayerSeasonStats(player_id=99, team_id=99, season_id=99, gp=82, pts=25.0)
    created = repo.upsert_player_stats(stats)
    assert created.id is not None
    assert created.pts == 25.0

    # Update
    stats2 = PlayerSeasonStats(player_id=99, team_id=99, season_id=99, gp=82, pts=27.0)
    updated = repo.upsert_player_stats(stats2)
    assert updated.id == created.id
    assert updated.pts == 27.0

    found = repo.find_by_season(99)
    assert len(found) == 1
    assert found[0].pts == 27.0


def test_game_log_create(db_session: Session):
    season = Season(id=200, season_label="2023-24-test-game")
    team = Team(id=200, abbreviation="LAL-TEST", full_name="Test Lakers", city="LA", conference="West", division="Pacific")
    player = Player(id=200, full_name="Test Gamer")
    db_session.add_all([season, team, player])
    db_session.commit()

    from app.models import PlayerGameLog
    log = PlayerGameLog(
        player_id=200,
        team_id=200,
        season_id=200,
        game_id="0022300001",
        game_date=date(2023, 10, 24),
        matchup="LAL @ DEN",
        wl="L",
        min=35,
        pts=21,
        plus_minus=-8,
    )
    db_session.add(log)
    db_session.commit()

    fetched = db_session.get(PlayerGameLog, log.id)
    assert fetched is not None
    assert fetched.pts == 21
    assert fetched.game_id == "0022300001"


def test_player_shot_create(db_session: Session):
    season = Season(id=201, season_label="2023-24-test-shot")
    player = Player(id=201, full_name="Test Shooter")
    db_session.add_all([season, player])
    db_session.commit()

    from app.models import PlayerShot
    shot = PlayerShot(
        player_id=201,
        season_id=201,
        game_id="0022300001",
        period=1,
        minutes_remaining=10,
        seconds_remaining=30,
        shot_distance=15.0,
        loc_x=100.0,
        loc_y=50.0,
        shot_attempted_flag=True,
        shot_made_flag=True,
    )
    db_session.add(shot)
    db_session.commit()

    fetched = db_session.get(PlayerShot, shot.id)
    assert fetched is not None
    assert fetched.shot_made_flag is True
    assert fetched.loc_x == 100.0
