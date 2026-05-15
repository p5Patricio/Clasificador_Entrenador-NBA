const RAW_API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export const API_ORIGIN = RAW_API_BASE.replace(/\/api\/v1\/?$/, "").replace(/\/$/, "");
export const API_BASE = `${API_ORIGIN}/api/v1`;

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

async function postJson<T>(url: string, payload: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

export type HealthResponse = {
  status: string;
  db_connected: boolean;
};

export type Team = {
  id: number;
  abbreviation: string;
  full_name: string;
  city: string;
  conference: string;
  division: string;
  players: number;
};

export type StatDelta = {
  stat: string;
  player_value: number;
  cluster_avg: number;
  diff: number;
  percent_diff: number;
};

export type PlayerAnalysisResponse = {
  player_name: string;
  team?: string | null;
  cluster_id?: number | null;
  comparison: StatDelta[];
};

export type PlayerProfileResponse = {
  player_id: number;
  full_name: string;
  team_id?: number | null;
  team_abbreviation?: string | null;
  team_name?: string | null;
  height?: string | null;
  weight?: string | null;
  height_cm?: number | null;
  weight_kg?: number | null;
  birthdate?: string | null;
  age?: number | null;
  headshot_url?: string | null;
};

export type PlayerShot = {
  x: number;
  y: number;
  made: boolean;
  action_type?: string | null;
  shot_zone_basic?: string | null;
  shot_distance?: number | null;
};

export type PlayerShotsResponse = {
  season?: string;
  season_type?: string;
  player_name?: string;
  season_id?: number;
  attempts: number;
  makes: number;
  shots: PlayerShot[];
};

export type PlayerRadarsResponse = {
  player_name: string;
  seasons: Array<{
    season_id: number;
    season_label?: string | null;
    stats: Record<string, number>;
  }>;
};

export type PlayerGameLogsResponse = {
  player_name: string;
  season_id: number;
  games: Array<{
    game_id: string;
    game_date?: string | null;
    matchup?: string | null;
    wl?: string | null;
    min: number;
    pts: number;
    reb: number;
    ast: number;
    stl: number;
    blk: number;
    plus_minus: number;
  }>;
};

export type PlayerAdvancedStatsResponse = {
  player_name: string;
  season_id: number;
  per?: number | null;
  ts_pct?: number | null;
  usg_pct?: number | null;
  ws?: number | null;
  bpm?: number | null;
  vorp?: number | null;
};

export type PlayerSimilaritiesResponse = {
  player_name: string;
  season_id: number;
  players: Array<{
    player_id: number;
    player_name: string;
    similarity_score: number;
  }>;
};

export type DataUpdateResponse = {
  season: string;
  file?: string | null;
  rows_processed: number;
  players_inserted: number;
  players_updated: number;
  players_failed: number;
  etl_status: string;
};

export async function getHealth() {
  return getJson<HealthResponse>(`${API_BASE}/health`);
}

export async function initCluster(payload: { season_id: number; k: number }) {
  const qs = new URLSearchParams({
    season_id: String(payload.season_id),
    k: String(payload.k),
  });
  return postJson<{ season_id: number; k: number; players: number; clusters: number; roles: Record<string, string> }>(
    `${API_BASE}/cluster/init?${qs.toString()}`,
    {},
  );
}

export async function getTeams() {
  return getJson<Team[]>(`${API_BASE}/teams`);
}

export async function getPlayers(team?: string, seasonId?: number) {
  const qs = new URLSearchParams();
  if (team) qs.set("team", team);
  if (seasonId) qs.set("season_id", String(seasonId));
  return getJson<string[]>(`${API_BASE}/players${qs.toString() ? `?${qs.toString()}` : ""}`);
}

export async function getPlayerAnalysis(name: string, seasonId?: number) {
  const qs = new URLSearchParams();
  if (seasonId) qs.set("season_id", String(seasonId));
  return getJson<PlayerAnalysisResponse>(
    `${API_BASE}/player/${encodeURIComponent(name)}/analysis${qs.toString() ? `?${qs.toString()}` : ""}`,
  );
}

export function getPlayerReportUrl(name: string, seasonId?: number) {
  const qs = new URLSearchParams();
  if (seasonId) qs.set("season_id", String(seasonId));
  return `${API_BASE}/player/${encodeURIComponent(name)}/report${qs.toString() ? `?${qs.toString()}` : ""}`;
}

export async function updateDataset(payload: { season: string; season_type?: string; min_minutes?: number; filepath?: string }) {
  return postJson<DataUpdateResponse>(`${API_BASE}/data/update`, payload);
}

export async function getPlayerProfile(name: string) {
  return getJson<PlayerProfileResponse>(`${API_BASE}/player/${encodeURIComponent(name)}/profile`);
}

export async function getPlayerShots(name: string, seasonId?: number, season?: string, seasonType?: string) {
  const qs = new URLSearchParams();
  if (seasonId) qs.set("season_id", String(seasonId));
  if (season) qs.set("season", season);
  if (seasonType) qs.set("season_type", seasonType);
  return getJson<PlayerShotsResponse>(`${API_BASE}/player/${encodeURIComponent(name)}/shots${qs.toString() ? `?${qs.toString()}` : ""}`);
}

export async function getPlayerRadars(name: string) {
  return getJson<PlayerRadarsResponse>(`${API_BASE}/player/${encodeURIComponent(name)}/radars`);
}

export async function getPlayerGameLogs(name: string, seasonId: number) {
  return getJson<PlayerGameLogsResponse>(`${API_BASE}/player/${encodeURIComponent(name)}/game-logs?season_id=${seasonId}`);
}

export async function getPlayerAdvancedStats(name: string, seasonId: number) {
  return getJson<PlayerAdvancedStatsResponse>(`${API_BASE}/player/${encodeURIComponent(name)}/advanced?season_id=${seasonId}`);
}

export async function getPlayerSimilarities(name: string, seasonId: number) {
  return getJson<PlayerSimilaritiesResponse>(`${API_BASE}/player/${encodeURIComponent(name)}/similar?season_id=${seasonId}`);
}
