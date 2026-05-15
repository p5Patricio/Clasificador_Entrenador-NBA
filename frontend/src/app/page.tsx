"use client";

import { FormEvent, useEffect, useState } from "react";
import Link from "next/link";
import { getHealth, initCluster, updateDataset, type HealthResponse } from "@/lib/api";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState(false);
  const [season, setSeason] = useState("2024-25");
  const [filepath, setFilepath] = useState("nba_active_player_stats_2024-25_Regular_Season_100min.xlsx");
  const [seasonId, setSeasonId] = useState(1);
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        setLoadingHealth(true);
        setHealth(await getHealth());
      } catch (err: unknown) {
        setError(errorMessage(err));
      } finally {
        setLoadingHealth(false);
      }
    };
    fetchHealth();
  }, []);

  const onUpdateDataset = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await updateDataset({
        season,
        season_type: "Regular Season",
        min_minutes: 100,
        filepath,
      });
      setMessage(
        `ETL ${result.etl_status}: ${result.rows_processed} filas, ${result.players_inserted} insertados, ${result.players_updated} actualizados, ${result.players_failed} fallidos.`,
      );
      setHealth(await getHealth());
    } catch (err: unknown) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const onInitCluster = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const result = await initCluster({ season_id: seasonId, k });
      setMessage(`Clustering inicializado: ${result.players} jugadores en ${result.clusters} clusters.`);
    } catch (err: unknown) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-8 rounded-2xl border border-blue-900/40 bg-gradient-to-br from-slate-900/60 to-blue-950/50 p-8 shadow-xl">
        <h1 className="mb-2 text-3xl font-semibold text-sky-400">NBA Analyzer</h1>
        <p className="text-sm text-slate-300">
          Plataforma de análisis NBA con datos históricos, similitud de jugadores, advanced stats y endpoints DB-backed.
        </p>
      </div>

      <section className="mb-8 rounded-xl border border-blue-900/40 bg-slate-900/60 p-6 shadow-lg">
        <h2 className="mb-3 text-xl font-semibold text-white">Estado del backend</h2>
        {loadingHealth ? (
          <p>Comprobando API…</p>
        ) : health ? (
          <p>
            status: <span className="font-mono text-sky-400">{health.status}</span> | db:{" "}
            <span className={health.db_connected ? "text-emerald-400" : "text-red-400"}>
              {health.db_connected ? "conectada" : "desconectada"}
            </span>
          </p>
        ) : (
          <p>No se pudo obtener el estado todavía.</p>
        )}
      </section>

      <section className="mb-8 rounded-xl border border-blue-900/40 bg-slate-900/60 p-6 shadow-lg">
        <h2 className="mb-4 text-xl font-semibold text-white">Actualizar dataset</h2>
        <form className="space-y-4" onSubmit={onUpdateDataset}>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block text-sm font-medium text-slate-200">
              Temporada
              <input
                type="text"
                className="mt-1 w-full rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100"
                value={season}
                onChange={(event) => setSeason(event.target.value)}
              />
            </label>
            <label className="block text-sm font-medium text-slate-200">
              Archivo Excel
              <input
                type="text"
                className="mt-1 w-full rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100"
                value={filepath}
                onChange={(event) => setFilepath(event.target.value)}
              />
            </label>
          </div>
          <button type="submit" className="rounded bg-emerald-600 px-4 py-2 text-white hover:bg-emerald-500 disabled:opacity-60" disabled={busy}>
            Ejecutar ETL
          </button>
        </form>
      </section>

      <section className="mb-8 rounded-xl border border-blue-900/40 bg-slate-900/60 p-6 shadow-lg">
        <h2 className="mb-4 text-xl font-semibold text-white">Inicializar clustering</h2>
        <form className="flex flex-wrap items-end gap-4" onSubmit={onInitCluster}>
          <label className="block text-sm font-medium text-slate-200">
            Season ID
            <input
              type="number"
              className="mt-1 w-36 rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100"
              min={1}
              value={seasonId}
              onChange={(event) => setSeasonId(Number(event.target.value))}
            />
          </label>
          <label className="block text-sm font-medium text-slate-200">
            K
            <input
              type="number"
              className="mt-1 w-24 rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100"
              min={2}
              max={12}
              value={k}
              onChange={(event) => setK(Number(event.target.value))}
            />
          </label>
          <button type="submit" className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 disabled:opacity-60" disabled={busy}>
            Inicializar
          </button>
        </form>
      </section>

      {message && <p className="mb-6 rounded border border-emerald-900/40 bg-emerald-950/40 p-3 text-emerald-300">{message}</p>}
      {error && <p className="mb-6 rounded border border-red-900/40 bg-red-950/40 p-3 text-red-300">{error}</p>}

      <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-6 shadow-lg">
        <h2 className="mb-3 text-xl font-semibold text-white">Navegación</h2>
        <Link href="/teams" className="text-sky-400 hover:text-sky-300">
          Ver equipos y jugadores
        </Link>
      </section>
    </div>
  );
}
