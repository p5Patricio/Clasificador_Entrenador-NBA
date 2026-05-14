# Phase 2 Data Engine + API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Phase 2 by making historical NBA data ingestion testable and exposing game logs, shots, advanced stats, player similarity, and team Elo through `/api/v1` endpoints.

**Architecture:** Keep the CLI thin. Move row mapping, calculations, query logic, and response composition into testable builders, repositories, and services. Routers only validate HTTP input and delegate to services.

**Tech Stack:** FastAPI, SQLModel, pytest, pandas, nba_api, optional BeautifulSoup/pandas `read_html` for Basketball Reference ingestion.

---

## File Structure

- Modify: `backend/app/etl/backfill.py` — keep CLI orchestration; delegate row mapping to helpers.
- Create: `backend/app/etl/builders.py` — pure functions for `PlayerGameLog`, `PlayerShot`, and advanced stat row mapping.
- Create: `backend/app/repositories/historical.py` — DB queries/upserts for Phase 2 entities.
- Create: `backend/app/services/historical.py` — player/game-log/shot/advanced/similarity/Elo read services.
- Create: `backend/app/services/elo.py` — Elo calculation logic.
- Create: `backend/app/services/similarity.py` — player similarity calculation logic.
- Modify: `backend/app/schemas.py` — response DTOs for new endpoints.
- Modify: `backend/app/api/deps.py` — DI provider for historical service/repository.
- Create: `backend/app/api/v1/historical.py` — new `/api/v1` endpoints.
- Modify: `backend/app/main.py` — include historical router.
- Create: `backend/tests/test_etl_builders.py` — RED/GREEN tests for row mapping.
- Create: `backend/tests/test_historical_repository.py` — repository tests.
- Create: `backend/tests/test_historical_services.py` — service tests.
- Create: `backend/tests/test_historical_api.py` — endpoint tests.

---

## Task 1: Extract Testable ETL Builders

**Files:**
- Create: `backend/app/etl/builders.py`
- Modify: `backend/app/etl/backfill.py`
- Test: `backend/tests/test_etl_builders.py`

- [ ] **Step 1: Write failing tests for shot row mapping**

Create tests that build a `PlayerShot` from a pandas-like row and assert: game id, event id, period, clock, zones, distance, coordinates, made flag, and parsed date.

Run: `cd backend; pytest tests/test_etl_builders.py::test_build_player_shot_from_row_maps_nba_api_columns -v`
Expected: FAIL because `app.etl.builders` does not exist.

- [ ] **Step 2: Implement minimal `build_player_shot_from_row`**

Move the inline `PlayerShot(...)` construction out of `backfill_shots()` into `builders.py`.

- [ ] **Step 3: Refactor `backfill_shots()` to use the builder**

Behavior must stay identical. No feature expansion yet.

- [ ] **Step 4: Run focused tests**

Run: `cd backend; pytest tests/test_etl_builders.py -v`
Expected: PASS.

---

## Task 2: Add Historical Repository

**Files:**
- Create: `backend/app/repositories/historical.py`
- Test: `backend/tests/test_historical_repository.py`

- [ ] **Step 1: Write failing tests for reads**

Cover:
- list player game logs by player + season
- list player shots by player + season
- get player advanced stats by player + season
- list similar players by player + season
- list team Elo by team + season

Expected: FAIL because `HistoricalRepository` does not exist.

- [ ] **Step 2: Implement minimal repository methods**

Use SQLModel `select()` and existing models. Keep filtering explicit and boring.

- [ ] **Step 3: Run repository tests**

Run: `cd backend; pytest tests/test_historical_repository.py -v`
Expected: PASS.

---

## Task 3: Add Historical DTOs and Service

**Files:**
- Modify: `backend/app/schemas.py`
- Create: `backend/app/services/historical.py`
- Test: `backend/tests/test_historical_services.py`

- [ ] **Step 1: Write failing service tests**

Cover service response shapes for:
- game logs
- DB-backed shots
- advanced stats
- similar players
- Elo timeline

Expected: FAIL because schemas/service do not exist.

- [ ] **Step 2: Add DTOs**

Add response models with stable API fields; do not expose raw SQLModel objects.

- [ ] **Step 3: Implement service mapping**

Service converts DB models into DTOs and raises `HTTPException(404)` for missing player/team where appropriate.

- [ ] **Step 4: Run service tests**

Run: `cd backend; pytest tests/test_historical_services.py -v`
Expected: PASS.

---

## Task 4: Add API Endpoints

**Files:**
- Modify: `backend/app/api/deps.py`
- Create: `backend/app/api/v1/historical.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_historical_api.py`

- [ ] **Step 1: Write failing API tests**

Endpoints:
- `GET /api/v1/player/{player_name}/game-logs?season_id=...`
- `GET /api/v1/player/{player_name}/shots?season_id=...`
- `GET /api/v1/player/{player_name}/advanced?season_id=...`
- `GET /api/v1/player/{player_name}/similar?season_id=...`
- `GET /api/v1/teams/{team_abbr}/elo?season_id=...`

Expected: FAIL with 404 route not found.

- [ ] **Step 2: Wire dependency provider**

Add `get_historical_repo()` and `get_historical_service()` in `backend/app/api/deps.py`.

- [ ] **Step 3: Implement router**

Router delegates only. No SQL in route handlers.

- [ ] **Step 4: Include router in app**

Register in `backend/app/main.py` with prefix `/api/v1`.

- [ ] **Step 5: Run API tests**

Run: `cd backend; pytest tests/test_historical_api.py -v`
Expected: PASS.

---

## Task 5: Advanced Stats Ingestion Skeleton

**Files:**
- Modify: `backend/app/etl/builders.py`
- Modify: `backend/app/etl/backfill.py`
- Test: `backend/tests/test_etl_builders.py`

- [ ] **Step 1: Write failing tests for Basketball Reference row mapping**

Map Basketball Reference columns to `PlayerAdvancedStats`: PER, TS%, USG%, WS, BPM, VORP.

- [ ] **Step 2: Implement pure builder**

Keep network scraping separate from row mapping.

- [ ] **Step 3: Add CLI flag only after builder is green**

Add advanced stats backfill entrypoint without making tests hit the network.

---

## Task 6: Elo Calculation Service

**Files:**
- Create: `backend/app/services/elo.py`
- Test: `backend/tests/test_elo_service.py`

- [ ] **Step 1: Write failing tests for Elo math**

Cover expected probability, winner rating increase, loser rating decrease, and deterministic order by game date.

- [ ] **Step 2: Implement minimal Elo logic**

Use default rating `1500`, default `k_factor=20`.

- [ ] **Step 3: Persist via historical repository**

Only after pure Elo math passes.

---

## Task 7: Player Similarity Service

**Files:**
- Create: `backend/app/services/similarity.py`
- Test: `backend/tests/test_similarity_service.py`

- [ ] **Step 1: Write failing tests for similarity scoring**

Use tiny fixture stats for 3 players. Assert closest player ranks first.

- [ ] **Step 2: Implement normalized vector similarity**

Start with cosine similarity over core per-game features: points, rebounds, assists, steals, blocks, shooting percentages.

- [ ] **Step 3: Persist top-N similarities**

Use existing `PlayerSimilarity` model.

---

## Task 8: Full Verification

**Files:**
- All modified files

- [ ] **Step 1: Run focused tests**

Run:
`cd backend; pytest tests/test_etl_builders.py tests/test_historical_repository.py tests/test_historical_services.py tests/test_historical_api.py tests/test_elo_service.py tests/test_similarity_service.py -v`

- [ ] **Step 2: Run full suite**

Run: `cd backend; pytest -v`

- [ ] **Step 3: Manual smoke only after tests pass**

Start API and hit the new endpoints with seeded test/dev data.

---

## Review Workload Forecast

- Estimated changed lines: 700–1100.
- Chained PRs recommended: Yes.
- 400-line budget risk: High.
- Decision needed before apply: Yes.

Recommended delivery: split into chained work units:
1. ETL builders + historical repository.
2. Historical service + API endpoints.
3. Advanced stats ingestion.
4. Elo + similarity calculations.
