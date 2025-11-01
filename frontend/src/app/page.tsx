"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHealth, initCluster, updateDataset } from "@/lib/api";

export default function Home() {
  const [health, setHealth] = useState<{ status: string; initialized: boolean } | null>(null);
  const [loadingHealth, setLoadingHealth] = useState(false);
  const [initLoading, setInitLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [filepath, setFilepath] = useState("nba_active_player_stats_2023-24_Regular_Season_100min.xlsx");
  const [k, setK] = useState(5);
  // Actualizar dataset desde NBA API
  const [season, setSeason] = useState<string>(""); // vacío = inferir temporada actual en backend
  const [minMinutes, setMinMinutes] = useState<number>(100);
  const [updatingData, setUpdatingData] = useState(false);
  const [updateMsg, setUpdateMsg] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        setLoadingHealth(true);
        const data = await getHealth();
        setHealth(data);
      } catch (e: any) {
        setError(e?.message ?? String(e));
      } finally {
        setLoadingHealth(false);
      }
    };
    fetchHealth();
  }, []);

  const onInit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setMessage(null);
    try {
      setInitLoading(true);
      const res = await initCluster({ filepath, k });
      setMessage(`${res.message}. Jugadores: ${res.players}, Clústeres: ${res.clusters}`);
      // refrescar health
      const h = await getHealth();
      setHealth(h);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setInitLoading(false);
    }
  };

  const onUpdateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setUpdateMsg(null);
    try {
      setUpdatingData(true);
      const res = await updateDataset({ season: season || undefined, min_minutes: minMinutes, auto_init: true });
      setUpdateMsg(`${res.message}. Temporada: ${res.season}. Archivo: ${res.file}. Reinicializado: ${res.initialized}`);
      const h = await getHealth();
      setHealth(h);
      // Si auto_init true, también actualizamos el mensaje de init
      setMessage(`Modelo inicializado desde NBA API. Temporada ${res.season}`);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setUpdatingData(false);
    }
  };

  return (
      <div className="mx-auto max-w-5xl px-0 py-0">
        <div className="mb-8 rounded-2xl border border-blue-900/40 bg-gradient-to-br from-slate-900/60 to-blue-950/50 p-8 shadow-xl">
          <h1 className="mb-2 text-3xl font-semibold text-sky-400">NBA Analyzer</h1>
          <p className="text-sm text-slate-300">Clustering de jugadores NBA, análisis comparativo y reportes PDF.</p>
        </div>

        <section className="mb-8 rounded-xl border border-blue-900/40 bg-slate-900/60 p-6 shadow-lg">
          <h2 className="mb-3 text-xl font-semibold text-white">Estado del backend</h2>
          {loadingHealth ? (
            <p>Comprobando /health…</p>
          ) : health ? (
            <p>
              status: <span className="font-mono text-sky-400">{health.status}</span> | initialized:{" "}
              <span className="font-mono">{String(health.initialized)}</span>
            </p>
          ) : (
            <p>No se pudo obtener el estado aún.</p>
          )}
        </section>

        <section className="mb-8 rounded-xl border border-blue-900/40 bg-slate-900/60 p-6 shadow-lg">
          <h2 className="mb-4 text-xl font-semibold text-white">Inicializar Clustering</h2>
          <form className="space-y-4" onSubmit={onInit}>
            <div>
              <label className="block text-sm font-medium text-slate-200">Ruta del Excel</label>
              <input
                type="text"
                className="mt-1 w-full rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100 placeholder:text-slate-500"
                value={filepath}
                onChange={(e) => setFilepath(e.target.value)}
                placeholder="nba_active_player_stats_...xlsx"
              />
              <p className="mt-1 text-xs text-slate-400">Si el archivo está en la raíz del repo, puedes usar el nombre tal cual.</p>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-200">K (número de clústeres)</label>
              <input
                type="number"
                className="mt-1 w-40 rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100"
                value={k}
                min={2}
                max={12}
                onChange={(e) => setK(Number(e.target.value))}
              />
            </div>
            <button
              type="submit"
              className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 disabled:opacity-60"
              disabled={initLoading}
            >
              {initLoading ? "Inicializando…" : "Inicializar"}
            </button>
          </form>
          {message && <p className="mt-3 text-green-400">{message}</p>}
          {error && <p className="mt-3 text-red-400">{error}</p>}
        </section>

        <section className="mb-8 rounded-xl border border-blue-900/40 bg-slate-900/60 p-6 shadow-lg">
          <h2 className="mb-4 text-xl font-semibold text-white">Actualizar dataset desde NBA API</h2>
          <form className="space-y-4" onSubmit={onUpdateDataset}>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-sm font-medium text-slate-200">Temporada</label>
                <input
                  type="text"
                  className="mt-1 w-full rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100 placeholder:text-slate-500"
                  value={season}
                  onChange={(e) => setSeason(e.target.value)}
                  placeholder="Ej: 2024-25 (vacío = detectar automáticamente)"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-200">Minutos mínimos</label>
                <input
                  type="number"
                  className="mt-1 w-full rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100"
                  value={minMinutes}
                  min={0}
                  onChange={(e) => setMinMinutes(Number(e.target.value))}
                />
              </div>
            </div>
            <button
              type="submit"
              className="rounded bg-emerald-600 px-4 py-2 text-white hover:bg-emerald-500 disabled:opacity-60"
              disabled={updatingData}
            >
              {updatingData ? "Actualizando…" : "Actualizar y reinicializar"}
            </button>
          </form>
          {updateMsg && <p className="mt-3 text-green-400">{updateMsg}</p>}
          {error && <p className="mt-3 text-red-400">{error}</p>}
        </section>

        <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-6 shadow-lg">
          <h2 className="mb-3 text-xl font-semibold text-white">Navegación</h2>
          <ul className="list-inside list-disc space-y-2">
            <li>
              <Link href="/teams" className="text-sky-400 hover:text-sky-300">
                Ver equipos y jugadores
              </Link>
            </li>
          </ul>
        </section>
      </div>
  );
}
