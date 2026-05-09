from fastapi import APIRouter
from sqlalchemy import text
from app.deps import engine
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    db_connected = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_connected = True
    except Exception:
        db_connected = False
    return HealthResponse(status="ok", db_connected=db_connected)
