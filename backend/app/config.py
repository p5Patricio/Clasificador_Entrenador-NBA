from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://nba:nba_pass@localhost:5432/nba_platform"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    DEBUG: bool = False
    ETL_BATCH_SIZE: int = 100

    class Config:
        env_file = ".env"


settings = Settings()
