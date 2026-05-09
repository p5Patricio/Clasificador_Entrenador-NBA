from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.v1 import health, teams, players, cluster, data, player_extras

app = FastAPI(
    title="NBA Analytics Platform API",
    version="0.2.0",
    description="Backend API for NBA player and team analytics",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Versioned API routers
@app.get("/")
def root():
    return {
        "message": "NBA Analytics Platform API",
        "version": "0.2.0",
        "docs": "/docs",
    }

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(teams.router, prefix="/api/v1", tags=["teams"])
app.include_router(players.router, prefix="/api/v1", tags=["players"])
app.include_router(cluster.router, prefix="/api/v1", tags=["cluster"])
app.include_router(data.router, prefix="/api/v1", tags=["data"])
app.include_router(player_extras.router, prefix="/api/v1", tags=["player-extras"])
