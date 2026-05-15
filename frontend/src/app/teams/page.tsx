"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getPlayers, getTeams, type Team } from "@/lib/api";

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function TeamsPage() {
  const [teams, setTeams] = useState<Team[] | null>(null);
  const [selectedTeam, setSelectedTeam] = useState<string>("");
  const [seasonId, setSeasonId] = useState<number>(1);
  const [players, setPlayers] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setTeams(await getTeams());
      } catch (err: unknown) {
        setError(errorMessage(err));
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
      setPlayers(await getPlayers(abbr, seasonId));
    } catch (err: unknown) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl">
      <h1 className="mb-2 text-3xl font-semibold text-sky-400">Equipos y jugadores</h1>
      <p className="mb-6 text-sm text-slate-300">
        Elegí un equipo para listar jugadores de una temporada. Si no aparecen jugadores, revisá que el Season ID coincida con la DB.
      </p>

      <label className="mb-6 block max-w-xs text-sm font-medium text-slate-200">
        Season ID
        <input
          type="number"
          min={1}
          value={seasonId}
          onChange={(event) => setSeasonId(Number(event.target.value))}
          className="mt-1 w-full rounded border border-blue-900/40 bg-slate-950/60 px-3 py-2 text-slate-100"
        />
      </label>

      {error && <p className="mb-4 rounded border border-red-900/40 bg-red-950/40 p-3 text-red-300">{error}</p>}

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
          <h2 className="mb-3 text-xl font-semibold text-white">Equipos</h2>
          {loading && !teams && <p>Cargando…</p>}
          {teams && (
            <ul className="max-h-[60vh] space-y-2 overflow-auto pr-2">
              {teams.map((team) => (
                <li key={team.abbreviation}>
                  <button
                    onClick={() => onSelectTeam(team.abbreviation)}
                    className={`w-full rounded border border-blue-900/40 bg-slate-950/40 px-3 py-2 text-left hover:bg-slate-900 ${
                      selectedTeam === team.abbreviation ? "bg-slate-900" : ""
                    }`}
                  >
                    <span className="font-semibold text-sky-400">{team.abbreviation}</span>
                    <span className="ml-2 text-sm text-slate-400">({team.players} jugadores)</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
          <h2 className="mb-3 text-xl font-semibold text-white">Jugadores {selectedTeam && `(${selectedTeam})`}</h2>
          {loading && selectedTeam && !players && <p>Cargando jugadores…</p>}
          {!selectedTeam && <p className="text-slate-400">Seleccioná un equipo.</p>}
          {players && players.length === 0 && <p>No hay jugadores listados para ese equipo/temporada.</p>}
          {players && players.length > 0 && (
            <ul className="max-h-[60vh] space-y-2 overflow-auto pr-2">
              {players.map((player) => (
                <li key={player}>
                  <Link className="block rounded border border-blue-900/40 bg-slate-950/40 px-3 py-2 hover:bg-slate-900" href={`/player/${encodeURIComponent(player)}`}>
                    {player}
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
