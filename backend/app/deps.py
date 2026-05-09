from contextlib import contextmanager
from sqlmodel import Session, create_engine, SQLModel
from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)


def get_db_session():
    """FastAPI dependency that yields a database session."""
    with Session(engine) as session:
        yield session


@contextmanager
def get_db_context():
    """Synchronous context manager for database sessions (ETL, scripts)."""
    with Session(engine) as session:
        yield session


def create_db_and_tables():
    """Create all tables defined in SQLModel metadata."""
    SQLModel.metadata.create_all(engine)
