export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export type Team = {
  abbreviation: string;
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
  player_id?: number | null;
  team?: string | null;
  cluster_id: number;
  cluster_role: string;
  headshot_url?: string | null;
  comparison: StatDelta[];
  weak_areas: string[];
  projected_stats: Record<string, number>;
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
  season: string;
  season_type: string;
  attempts: number;
  makes: number;
  shots: PlayerShot[];
};

export type PlayerRadarsResponse = {
  player_name: string;
  cluster_id: number;
  cluster_role: string;
  categories: Array<{
    name: string;
    stats: string[];
    labels: string[];
    series: {
      player: number[];
      cluster_avg: number[];
      projected: number[];
    };
  }>;
};

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Health error: ${res.status}`);
  return res.json() as Promise<{ status: string; initialized: boolean }>;
}

export async function initCluster(payload: { filepath: string; k: number; stats_columns?: string[] }) {
  const res = await fetch(`${API_BASE}/cluster/init`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ message: string; k: number; players: number; clusters: number; roles: Record<string, string> }>;
}

export async function getTeams() {
  const res = await fetch(`${API_BASE}/teams`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<Team[]>;
}

export async function getPlayers(team?: string) {
  const url = team ? `${API_BASE}/players?team=${encodeURIComponent(team)}` : `${API_BASE}/players`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<string[]>;
}

export async function getPlayerAnalysis(name: string) {
  const res = await fetch(`${API_BASE}/player/${encodeURIComponent(name)}/analysis`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PlayerAnalysisResponse>;
}

export function getPlayerReportUrl(name: string) {
  return `${API_BASE}/player/${encodeURIComponent(name)}/report`;
}

export async function updateDataset(payload: { season?: string; season_type?: string; min_minutes?: number; auto_init?: boolean }) {
  const res = await fetch(`${API_BASE}/data/update`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ message: string; season: string; season_type: string; file: string; initialized: boolean }>;
}

export async function getPlayerProfile(name: string) {
  const res = await fetch(`${API_BASE}/player/${encodeURIComponent(name)}/profile`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PlayerProfileResponse>;
}

export async function getPlayerShots(name: string, season?: string, season_type?: string) {
  const qs = new URLSearchParams();
  if (season) qs.set("season", season);
  if (season_type) qs.set("season_type", season_type);
  const url = `${API_BASE}/player/${encodeURIComponent(name)}/shots${qs.toString() ? `?${qs.toString()}` : ""}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PlayerShotsResponse>;
}

export async function getPlayerRadars(name: string) {
  const res = await fetch(`${API_BASE}/player/${encodeURIComponent(name)}/radars`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<PlayerRadarsResponse>;
}
