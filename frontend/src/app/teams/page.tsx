"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getTeams, getPlayers, type Team } from "@/lib/api";

export default function TeamsPage() {
  const [teams, setTeams] = useState<Team[] | null>(null);
  const [selectedTeam, setSelectedTeam] = useState<string>("");
  const [players, setPlayers] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const data = await getTeams();
        setTeams(data);
      } catch (e: any) {
        setError(e?.message ?? String(e));
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const onSelectTeam = async (abbr: string) => {
    setSelectedTeam(abbr);
    setPlayers(null);
    setError(null);
    try {
      setLoading(true);
      const list = await getPlayers(abbr);
      setPlayers(list);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
      <div className="mx-auto max-w-5xl px-0 py-0">
        <h1 className="mb-2 text-3xl font-semibold text-sky-400">Equipos y Jugadores</h1>
        <p className="mb-6 text-sm text-slate-300">Asegúrate de haber inicializado el clustering en la Home.</p>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
            <h2 className="mb-3 text-xl font-semibold text-white">Equipos</h2>
            {loading && !teams && <p>Cargando…</p>}
            {error && <p className="text-red-400">{error}</p>}
            {teams && (
              <ul className="max-h-[60vh] space-y-2 overflow-auto pr-2">
                {teams.map((t) => (
                  <li key={t.abbreviation}>
                    <button
                      onClick={() => onSelectTeam(t.abbreviation)}
                      className={`w-full rounded border border-blue-900/40 bg-slate-950/40 px-3 py-2 text-left hover:bg-slate-900 ${
                        selectedTeam === t.abbreviation ? "bg-slate-900" : ""
                      }`}
                    >
                      <span className="font-semibold text-sky-400">{t.abbreviation}</span>
                      <span className="ml-2 text-sm text-slate-400">({t.players} jugadores)</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
            <h2 className="mb-3 text-xl font-semibold text-white">Jugadores {selectedTeam && `(${selectedTeam})`}</h2>
            {loading && selectedTeam && !players && <p>Cargando jugadores…</p>}
            {!selectedTeam && <p className="text-slate-400">Selecciona un equipo.</p>}
            {players && players.length === 0 && <p>No hay jugadores listados para este equipo.</p>}
            {players && players.length > 0 && (
              <ul className="max-h-[60vh] space-y-2 overflow-auto pr-2">
                {players.map((p) => (
                  <li key={p}>
                    <Link
                      className="block rounded border border-blue-900/40 bg-slate-950/40 px-3 py-2 hover:bg-slate-900"
                      href={`/player/${encodeURIComponent(p)}`}
                    >
                      {p}
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>

        <div className="mt-6">
          <Link href="/" className="text-sky-400 hover:text-sky-300">
            ← Volver a Home
          </Link>
        </div>
      </div>
  );
}
