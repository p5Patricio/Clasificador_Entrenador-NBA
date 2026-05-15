"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  API_BASE,
  getPlayerAdvancedStats,
  getPlayerAnalysis,
  getPlayerGameLogs,
  getPlayerProfile,
  getPlayerRadars,
  getPlayerReportUrl,
  getPlayerShots,
  getPlayerSimilarities,
  type PlayerAdvancedStatsResponse,
  type PlayerAnalysisResponse,
  type PlayerGameLogsResponse,
  type PlayerProfileResponse,
  type PlayerRadarsResponse,
  type PlayerShotsResponse,
  type PlayerSimilaritiesResponse,
} from "@/lib/api";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatNumber(value?: number | null, digits = 2): string {
  return typeof value === "number" ? value.toFixed(digits) : "N/A";
}

export default function PlayerPage() {
  const params = useParams<{ name: string }>();
  const rawName = Array.isArray(params?.name) ? params.name.join("/") : params?.name ?? "";
  const playerName = decodeURIComponent(rawName);

  const [seasonId, setSeasonId] = useState(1);
  const [analysis, setAnalysis] = useState<PlayerAnalysisResponse | null>(null);
  const [profile, setProfile] = useState<PlayerProfileResponse | null>(null);
  const [radars, setRadars] = useState<PlayerRadarsResponse | null>(null);
  const [shots, setShots] = useState<PlayerShotsResponse | null>(null);
  const [advanced, setAdvanced] = useState<PlayerAdvancedStatsResponse | null>(null);
  const [similar, setSimilar] = useState<PlayerSimilaritiesResponse | null>(null);
  const [logs, setLogs] = useState<PlayerGameLogsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadCore = async () => {
    setLoading(true);
    setError(null);
    try {
      const [profileResult, radarsResult] = await Promise.all([
        getPlayerProfile(playerName).catch(() => null),
        getPlayerRadars(playerName).catch(() => null),
      ]);
      setProfile(profileResult);
      setRadars(radarsResult);
      if (radarsResult?.seasons?.[0]?.season_id) {
        setSeasonId(radarsResult.seasons[0].season_id);
      }
    } catch (err: unknown) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const loadSeasonData = async (targetSeasonId: number) => {
    setLoading(true);
    setError(null);
    try {
      const [analysisResult, shotsResult, advancedResult, similarResult, logsResult] = await Promise.all([
        getPlayerAnalysis(playerName, targetSeasonId).catch(() => null),
        getPlayerShots(playerName, targetSeasonId).catch(() => null),
        getPlayerAdvancedStats(playerName, targetSeasonId).catch(() => null),
        getPlayerSimilarities(playerName, targetSeasonId).catch(() => null),
        getPlayerGameLogs(playerName, targetSeasonId).catch(() => null),
      ]);
      setAnalysis(analysisResult);
      setShots(shotsResult);
      setAdvanced(advancedResult);
      setSimilar(similarResult);
      setLogs(logsResult);
    } catch (err: unknown) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadCore();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playerName]);

  useEffect(() => {
    void loadSeasonData(seasonId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playerName, seasonId]);

  const onSeasonSubmit = (event: FormEvent) => {
    event.preventDefault();
    void loadSeasonData(seasonId);
  };

  const headshotUrl = profile?.headshot_url || `${API_BASE}/player/${encodeURIComponent(playerName)}/headshot`;

  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-6 flex flex-wrap items-center gap-4">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={headshotUrl}
          alt={playerName}
          onError={(event) => {
            event.currentTarget.src = `${API_BASE}/player/${encodeURIComponent(playerName)}/headshot`;
          }}
          className="h-28 w-28 rounded-lg border border-blue-900/40 object-cover shadow"
        />
        <div>
          <h1 className="text-3xl font-semibold text-sky-400">{playerName}</h1>
          <p className="mt-1 text-sm text-slate-300">
            Equipo: <span className="font-mono">{analysis?.team ?? profile?.team_abbreviation ?? "N/A"}</span> · Cluster:{" "}
            <span className="font-mono">{analysis?.cluster_id ?? "N/A"}</span>
          </p>
          {profile && (
            <p className="mt-1 text-sm text-slate-400">
              Altura: {profile.height_cm ? `${profile.height_cm.toFixed(1)} cm` : "N/A"} · Peso:{" "}
              {profile.weight_kg ? `${profile.weight_kg.toFixed(1)} kg` : "N/A"} · Edad: {profile.age ? `${profile.age.toFixed(1)} años` : "N/A"}
            </p>
          )}
        </div>
      </div>

      <form className="mb-6 flex flex-wrap items-end gap-3 rounded-xl border border-blue-900/40 bg-slate-900/60 p-4" onSubmit={onSeasonSubmit}>
        <label className="text-sm font-medium text-slate-200">
          Season ID
          <input
            type="number"
            min={1}
            value={seasonId}
            onChange={(event) => setSeasonId(Number(event.target.value))}
            className="mt-1 block w-32 rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100"
          />
        </label>
        <button type="submit" className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 disabled:opacity-60" disabled={loading}>
          Cargar temporada
        </button>
        <a href={getPlayerReportUrl(playerName, seasonId)} target="_blank" rel="noreferrer" className="rounded border border-blue-900/40 px-4 py-2 text-sky-300 hover:bg-slate-900">
          Descargar PDF
        </a>
      </form>

      {loading && <p className="mb-4 text-slate-300">Cargando…</p>}
      {error && <p className="mb-4 rounded border border-red-900/40 bg-red-950/40 p-3 text-red-300">{error}</p>}

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
          <h2 className="mb-3 text-xl font-semibold text-white">Advanced stats</h2>
          {advanced ? (
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <Metric label="PER" value={formatNumber(advanced.per)} />
              <Metric label="TS%" value={formatNumber(advanced.ts_pct)} />
              <Metric label="USG%" value={formatNumber(advanced.usg_pct)} />
              <Metric label="WS" value={formatNumber(advanced.ws)} />
              <Metric label="BPM" value={formatNumber(advanced.bpm)} />
              <Metric label="VORP" value={formatNumber(advanced.vorp)} />
            </dl>
          ) : (
            <p className="text-slate-400">Sin advanced stats para esta temporada.</p>
          )}
        </section>

        <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
          <h2 className="mb-3 text-xl font-semibold text-white">Jugadores similares</h2>
          {similar?.players.length ? (
            <ol className="space-y-2">
              {similar.players.map((player) => (
                <li key={player.player_id} className="flex justify-between rounded border border-blue-900/30 bg-slate-950/40 px-3 py-2">
                  <span>{player.player_name}</span>
                  <span className="font-mono text-sky-300">{formatNumber(player.similarity_score, 4)}</span>
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-slate-400">Sin similitudes calculadas.</p>
          )}
        </section>
      </div>

      <section className="mt-6 rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
        <h2 className="mb-3 text-xl font-semibold text-white">Resumen por temporada</h2>
        {radars?.seasons.length ? (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {radars.seasons.map((season) => (
              <button
                key={season.season_id}
                onClick={() => setSeasonId(season.season_id)}
                className="rounded border border-blue-900/40 bg-slate-950/40 p-3 text-left hover:bg-slate-900"
              >
                <p className="font-semibold text-sky-300">{season.season_label ?? `Season ${season.season_id}`}</p>
                <p className="text-sm text-slate-300">
                  PTS {formatNumber(season.stats.pts)} · REB {formatNumber(season.stats.reb)} · AST {formatNumber(season.stats.ast)}
                </p>
              </button>
            ))}
          </div>
        ) : (
          <p className="text-slate-400">Sin temporadas cargadas.</p>
        )}
      </section>

      <section className="mt-6 rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
        <h2 className="mb-3 text-xl font-semibold text-white">Comparativa de cluster</h2>
        {analysis?.comparison.length ? (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-800/70">
                <tr>
                  <th className="px-3 py-2 text-left">Stat</th>
                  <th className="px-3 py-2 text-right">Jugador</th>
                  <th className="px-3 py-2 text-right">Cluster</th>
                  <th className="px-3 py-2 text-right">Dif</th>
                </tr>
              </thead>
              <tbody>
                {analysis.comparison.map((item) => (
                  <tr key={item.stat} className="border-t border-blue-900/30">
                    <td className="px-3 py-2">{item.stat}</td>
                    <td className="px-3 py-2 text-right">{formatNumber(item.player_value)}</td>
                    <td className="px-3 py-2 text-right">{formatNumber(item.cluster_avg)}</td>
                    <td className="px-3 py-2 text-right">{formatNumber(item.diff)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-slate-400">Sin comparativa para esta temporada.</p>
        )}
      </section>

      <section className="mt-6 rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
        <h2 className="mb-3 text-xl font-semibold text-white">Tiros DB-backed</h2>
        {shots ? (
          <p className="text-slate-300">
            Intentos: <span className="font-mono text-sky-300">{shots.attempts}</span> · Anotados:{" "}
            <span className="font-mono text-emerald-300">{shots.makes}</span>
          </p>
        ) : (
          <p className="text-slate-400">Sin tiros históricos cargados.</p>
        )}
      </section>

      <section className="mt-6 rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
        <h2 className="mb-3 text-xl font-semibold text-white">Game logs</h2>
        {logs?.games.length ? (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-800/70">
                <tr>
                  <th className="px-3 py-2 text-left">Fecha</th>
                  <th className="px-3 py-2 text-left">Matchup</th>
                  <th className="px-3 py-2 text-right">PTS</th>
                  <th className="px-3 py-2 text-right">REB</th>
                  <th className="px-3 py-2 text-right">AST</th>
                </tr>
              </thead>
              <tbody>
                {logs.games.map((game) => (
                  <tr key={game.game_id} className="border-t border-blue-900/30">
                    <td className="px-3 py-2">{game.game_date ?? "N/A"}</td>
                    <td className="px-3 py-2">{game.matchup ?? game.game_id}</td>
                    <td className="px-3 py-2 text-right">{game.pts}</td>
                    <td className="px-3 py-2 text-right">{game.reb}</td>
                    <td className="px-3 py-2 text-right">{game.ast}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-slate-400">Sin game logs para esta temporada.</p>
        )}
      </section>

      <div className="mt-6">
        <Link href="/teams" className="text-sky-400 hover:text-sky-300">
          ← Volver a equipos
        </Link>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-blue-900/30 bg-slate-950/40 p-3">
      <dt className="text-slate-400">{label}</dt>
      <dd className="font-mono text-lg text-sky-300">{value}</dd>
    </div>
  );
}
