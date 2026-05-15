# Phase 2 Product Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the completed Phase 2 backend into product-visible functionality by wiring real data update, Redis-backed caching, frontend consumption of historical endpoints, and technical cleanup.

**Architecture:** Keep backend mutations explicit and testable: `/data/update` delegates to an injectable service instead of containing ETL logic in the router. Cache is a thin dependency with a no-op fallback so tests do not require Redis. Frontend API types mirror backend DTOs, and UI reads Phase 2 data through `src/lib/api.ts`.

**Tech Stack:** FastAPI, SQLModel, pytest, redis-py, Pydantic Settings v2, Next.js 16 App Router, React 19, TypeScript.

---

## Task 1: Real `/api/v1/data/update`

**Files:**
- Create: `backend/app/services/data_update.py`
- Modify: `backend/app/api/deps.py`
- Modify: `backend/app/api/v1/data.py`
- Modify: `backend/app/schemas.py`
- Test: `backend/tests/test_data_update_api.py`

- [ ] Write a failing API test proving `/api/v1/data/update` calls an injectable service and returns inserted/updated/failed counts.
- [ ] Implement `DataUpdateService` around `ETLPipeline`.
- [ ] Wire FastAPI dependency.
- [ ] Make `DataUpdateResponse` match the real ETL summary.
- [ ] Run focused tests.

## Task 2: Redis cache wrapper and cached historical reads

**Files:**
- Create: `backend/app/services/cache.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/services/historical.py`
- Test: `backend/tests/test_cache_service.py`
- Test: `backend/tests/test_historical_services.py`

- [ ] Write failing unit tests for cache get/set/delete-prefix behavior using an in-memory fake.
- [ ] Implement cache wrapper with no-op behavior when Redis is unavailable.
- [ ] Cache expensive historical service reads.
- [ ] Invalidate historical cache after `/data/update`.
- [ ] Run focused tests.

## Task 3: Frontend Phase 2 API client and UI

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/app/player/[name]/page.tsx`
- Modify: `frontend/src/app/page.tsx`

- [ ] Add typed clients for game logs, advanced stats, similarities, team Elo, and DB-backed shots with `season_id`.
- [ ] Render advanced stats, similar players, game logs, and DB-backed shots on player page.
- [ ] Fix headshot fallback to use `API_BASE`.
- [ ] Remove stale 2023-24 default or make it a selectable explicit value.
- [ ] Run frontend lint/build after dependencies are available.

## Task 4: Pydantic v2 cleanup and final verification

**Files:**
- Modify: `backend/app/config.py`

- [ ] Replace class-based `Config` with `SettingsConfigDict`.
- [ ] Run full backend test suite.
- [ ] Run frontend lint/build.
- [ ] Commit by work unit.
