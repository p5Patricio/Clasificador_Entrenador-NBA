"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  getPlayerAnalysis,
  getPlayerReportUrl,
  type PlayerAnalysisResponse,
  getPlayerProfile,
  type PlayerProfileResponse,
  getPlayerShots,
  type PlayerShotsResponse,
  getPlayerRadars,
  type PlayerRadarsResponse,
} from "@/lib/api";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  LinearScale,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import { Radar, Scatter } from "react-chartjs-2";

// Registrar piezas del radar/escatter
ChartJS.register(RadialLinearScale, PointElement, LineElement, LinearScale, Filler, Tooltip, Legend);

export default function PlayerPage() {
  const params = useParams<{ name: string }>();
  const rawName = Array.isArray(params?.name) ? params.name.join("/") : params?.name ?? "";
  const playerName = decodeURIComponent(rawName);

  const [data, setData] = useState<PlayerAnalysisResponse | null>(null);
  const [profile, setProfile] = useState<PlayerProfileResponse | null>(null);
  const [shots, setShots] = useState<PlayerShotsResponse | null>(null);
  const [radars, setRadars] = useState<PlayerRadarsResponse | null>(null);
  const [catIndex, setCatIndex] = useState(0);

  const categories = useMemo(() => (radars?.categories || []).filter((c) => c.name !== "Global"), [radars]);

  useEffect(() => {
    // Reset index when categories change
    setCatIndex(0);
  }, [categories.length]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPhoto, setShowPhoto] = useState(false);

  // Etiquetas legibles en español
  const STAT_LABELS: Record<string, string> = useMemo(
    () => ({
      GP: "Partidos jugados",
      GS: "Partidos como titular",
      MIN: "Minutos",
      FGM: "Tiros de campo anotados",
      FGA: "Tiros de campo intentados",
      FG_PCT: "% de tiros de campo",
      FG3M: "Triples anotados",
      FG3A: "Triples intentados",
      FG3_PCT: "% de triples",
      FTM: "Tiros libres anotados",
      FTA: "Tiros libres intentados",
      FT_PCT: "% de tiros libres",
      OREB: "Rebotes ofensivos",
      DREB: "Rebotes defensivos",
      REB: "Rebotes",
      AST: "Asistencias",
      STL: "Robos",
      BLK: "Tapones",
      TOV: "Pérdidas",
      PF: "Faltas personales",
      PTS: "Puntos",
    }),
    []
  );

  const seasonTypeEs = (t?: string) => {
    if (!t) return "";
    const m: Record<string, string> = {
      "Regular Season": "Temporada Regular",
      Playoffs: "Playoffs",
      "Pre Season": "Pretemporada",
      "In Season Tournament": "Copa In-Season",
    };
    return m[t] || t;
  };

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const [res, prof, sc, rd] = await Promise.all([
          getPlayerAnalysis(playerName),
          getPlayerProfile(playerName).catch(() => null),
          getPlayerShots(playerName).catch(() => null),
          getPlayerRadars(playerName).catch(() => null),
        ]);
        setData(res);
        if (prof) setProfile(prof);
        if (sc) setShots(sc);
        if (rd) setRadars(rd);
      } catch (e: any) {
        setError(e?.message ?? String(e));
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [playerName]);

  const onDownloadPdf = () => {
    const url = getPlayerReportUrl(playerName);
    window.open(url, "_blank");
  };

  return (
    <div className="mx-auto max-w-6xl px-0 py-0">
      <div className="mb-4 flex items-center gap-4">
        {(data?.headshot_url || profile?.headshot_url) && (
          <img
            src={(data?.headshot_url || profile?.headshot_url) as string}
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).src = `${process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8001"}/player/${encodeURIComponent(playerName)}/headshot`;
            }}
            alt={playerName}
            title="Click para ampliar"
            onClick={() => setShowPhoto(true)}
            className="h-28 w-28 cursor-zoom-in rounded-lg border border-blue-900/40 object-cover shadow"
          />
        )}
        <div>
          <h1 className="text-3xl font-semibold text-sky-400">{playerName}</h1>
          {data && (
            <p className="mt-1 text-sm text-slate-300">
              Equipo: <span className="font-mono">{data.team ?? "N/A"}</span> · Clúster: <span className="font-mono">{data.cluster_id}</span> · Rol: <span className="font-semibold text-sky-300">{data.cluster_role}</span>
            </p>
          )}
          {profile && (
            <p className="mt-1 text-sm text-slate-300">
              Equipo (API): <span className="font-mono">{profile.team_abbreviation ?? "N/A"}</span> · Altura: {profile.height_cm ? `${profile.height_cm.toFixed(1)} cm` : "N/A"} · Peso: {profile.weight_kg ? `${profile.weight_kg.toFixed(1)} kg` : "N/A"} · Edad: {profile.age ? `${profile.age.toFixed(1)} años` : "N/A"}
            </p>
          )}
        </div>
      </div>

      {/* Modal para ampliar la foto */}
      {showPhoto && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={() => setShowPhoto(false)}>
          <img
            src={(data?.headshot_url || profile?.headshot_url) as string}
            alt={playerName}
            className="max-h-[85vh] max-w-[85vw] cursor-zoom-out rounded-xl border border-blue-900/50 shadow-2xl"
          />
        </div>
      )}

      {loading && <p>Cargando análisis…</p>}
      {error && <p className="text-red-400">{error}</p>}

      {data && (
        <div className="space-y-6">
          {/* Radar Global desde backend (mismas normalizaciones que PDF) */}
          <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
            <h2 className="mb-3 text-xl font-semibold text-white">Radar de rendimiento (Global)</h2>
            <div className="w-full overflow-x-auto">
              {(() => {
                const g = radars?.categories?.find((c) => c.name === "Global");
                const labels = g ? g.labels.map((l, i) => STAT_LABELS[g.stats[i]] || l) : data.comparison.map((c) => STAT_LABELS[c.stat] || c.stat);
                const playerValues = g ? g.series.player : [];
                const clusterValues = g ? g.series.cluster_avg : [];
                const projectedValues = g ? g.series.projected : [];

                const chartData = {
                  labels,
                  datasets: [
                    {
                      label: "Jugador",
                      data: playerValues,
                      backgroundColor: "rgba(34, 211, 238, 0.20)",
                      borderColor: "#22d3ee",
                      borderWidth: 3,
                      pointRadius: 2,
                      pointHoverRadius: 3,
                      pointBackgroundColor: "#22d3ee",
                      pointBorderColor: "#061018",
                    },
                    {
                      label: "Prom. Clúster",
                      data: clusterValues,
                      backgroundColor: "rgba(168, 85, 247, 0.12)",
                      borderColor: "#a855f7",
                      borderWidth: 2,
                      pointRadius: 1.5,
                      pointHoverRadius: 2.5,
                      pointBackgroundColor: "#a855f7",
                      pointBorderColor: "#0b0620",
                    },
                    {
                      label: "Proyección",
                      data: projectedValues,
                      backgroundColor: "rgba(59, 130, 246, 0.10)",
                      borderColor: "#3b82f6",
                      borderWidth: 2,
                      pointRadius: 1,
                      pointHoverRadius: 2,
                      pointBackgroundColor: "#3b82f6",
                      pointBorderColor: "#0b0620",
                      borderDash: [6, 4] as number[],
                    },
                  ],
                };

                const options = {
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { labels: { color: "#cbd5e1" } },
                    tooltip: { enabled: true },
                  },
                  scales: {
                    r: {
                      angleLines: { color: "rgba(30,58,138,0.3)" },
                      grid: { color: "rgba(30,58,138,0.3)" },
                      pointLabels: { color: "#cbd5e1", font: { size: 11 } },
                      suggestedMin: 0,
                      suggestedMax: 1,
                      ticks: {
                        showLabelBackdrop: false,
                        color: "#94a3b8",
                        backdropColor: "transparent",
                        stepSize: 0.25,
                      },
                    },
                  },
                };

                return (
                  <div className="h-[420px] min-w-[520px]">
                    <Radar data={chartData} options={options} />
                  </div>
                );
              })()}
            </div>
          </section>

          {/* Radares por categoría: Carrusel */}
          {categories.length > 0 && (
            <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-xl font-semibold text-white">Radares por categoría</h2>
                <div className="flex items-center gap-2">
                  <button
                    className="rounded-md border border-blue-900/40 bg-slate-800 px-3 py-1 text-slate-200 hover:bg-slate-700 disabled:opacity-50"
                    onClick={() => setCatIndex((i) => (i - 1 + categories.length) % categories.length)}
                    disabled={categories.length <= 1}
                    aria-label="Anterior"
                  >
                    ◀
                  </button>
                  <button
                    className="rounded-md border border-blue-900/40 bg-slate-800 px-3 py-1 text-slate-200 hover:bg-slate-700 disabled:opacity-50"
                    onClick={() => setCatIndex((i) => (i + 1) % categories.length)}
                    disabled={categories.length <= 1}
                    aria-label="Siguiente"
                  >
                    ▶
                  </button>
                </div>
              </div>

              {(() => {
                const cat = categories[catIndex];
                const labels = cat.labels.map((l, i) => STAT_LABELS[cat.stats[i]] || l);
                const chartData = {
                  labels,
                  datasets: [
                    {
                      label: "Jugador",
                      data: cat.series.player,
                      backgroundColor: "rgba(34, 211, 238, 0.20)",
                      borderColor: "#22d3ee",
                      borderWidth: 3,
                      pointRadius: 2,
                      pointHoverRadius: 3,
                      pointBackgroundColor: "#22d3ee",
                      pointBorderColor: "#061018",
                    },
                    {
                      label: "Prom. Clúster",
                      data: cat.series.cluster_avg,
                      backgroundColor: "rgba(168, 85, 247, 0.12)",
                      borderColor: "#a855f7",
                      borderWidth: 2,
                      pointRadius: 1.5,
                      pointHoverRadius: 2.5,
                      pointBackgroundColor: "#a855f7",
                      pointBorderColor: "#0b0620",
                    },
                    {
                      label: "Proyección",
                      data: cat.series.projected,
                      backgroundColor: "rgba(59, 130, 246, 0.10)",
                      borderColor: "#3b82f6",
                      borderWidth: 2,
                      pointRadius: 1,
                      pointHoverRadius: 2,
                      pointBackgroundColor: "#3b82f6",
                      pointBorderColor: "#0b0620",
                      borderDash: [6, 4] as number[],
                    },
                  ],
                };
                const options = {
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { labels: { color: "#cbd5e1" } } },
                  scales: {
                    r: {
                      angleLines: { color: "rgba(30,58,138,0.3)" },
                      grid: { color: "rgba(30,58,138,0.3)" },
                      pointLabels: { color: "#cbd5e1", font: { size: 12 } },
                      suggestedMin: 0,
                      suggestedMax: 1,
                      ticks: { showLabelBackdrop: false, color: "#94a3b8", backdropColor: "transparent", stepSize: 0.25 },
                    },
                  },
                };
                return (
                  <div className="rounded-lg border border-blue-900/30 bg-slate-900/40 p-3">
                    <h3 className="mb-2 text-lg font-semibold text-sky-300">{cat.name}</h3>
                    <div className="h-[560px]">
                      <Radar data={chartData} options={options} />
                    </div>
                  </div>
                );
              })()}

              {/* Dots */}
              {categories.length > 1 && (
                <div className="mt-4 flex justify-center gap-2">
                  {categories.map((_, i) => (
                    <button
                      key={i}
                      onClick={() => setCatIndex(i)}
                      className={`${i === catIndex ? 'bg-sky-400' : 'bg-slate-600'} h-2.5 w-2.5 rounded-full`}
                      aria-label={`Ir a categoría ${i + 1}`}
                    />
                  ))}
                </div>
              )}
            </section>
          )}

          {shots && (
            <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
              <h2 className="mb-3 text-xl font-semibold text-white">Mapa de tiros ({shots.season}, {seasonTypeEs(shots.season_type)})</h2>
              <div className="h-[480px] w-full">
                {(() => {
                  const made = shots.shots.filter((s) => s.made);
                  const missed = shots.shots.filter((s) => !s.made);

                  const data = {
                    datasets: [
                      {
                        label: "Encestados",
                        data: made.map((s) => ({ x: s.x, y: s.y })),
                        pointBackgroundColor: "#22c55e",
                        pointBorderColor: "#052e16",
                        pointRadius: 3.5,
                        pointStyle: "circle" as const,
                        showLine: false,
                      },
                      {
                        label: "Fallados",
                        data: missed.map((s) => ({ x: s.x, y: s.y })),
                        pointBackgroundColor: "transparent",
                        pointBorderColor: "#ef4444",
                        pointRadius: 4,
                        borderWidth: 1.5,
                        pointStyle: "cross" as const,
                        showLine: false,
                      },
                    ],
                  };

                  // Plugin para dibujar media cancha
                  const courtPlugin = {
                    id: "court-bg",
                    afterDraw(chart: any) {
                      const { ctx, scales } = chart;
                      const xs = scales.x;
                      const ys = scales.y;
                      if (!xs || !ys) return;
                      ctx.save();
                      ctx.lineWidth = 1;
                      ctx.strokeStyle = "rgba(148,163,184,0.25)";
                      const xL = xs.getPixelForValue(-250);
                      const xR = xs.getPixelForValue(250);
                      const yB = ys.getPixelForValue(-50);
                      const yT = ys.getPixelForValue(470);
                      ctx.strokeRect(xL, yT, xR - xL, yB - yT);
                      const zxL = xs.getPixelForValue(-80);
                      const zxR = xs.getPixelForValue(80);
                      const zyT = ys.getPixelForValue(140);
                      ctx.strokeRect(zxL, zyT, zxR - zxL, yB - zyT);
                      const cx = xs.getPixelForValue(0);
                      const cy = ys.getPixelForValue(140);
                      const r60 = Math.abs(xs.getPixelForValue(60) - xs.getPixelForValue(0));
                      ctx.beginPath();
                      ctx.arc(cx, cy, r60, 0, Math.PI, true);
                      ctx.stroke();
                      const cy0 = ys.getPixelForValue(0);
                      const rHoop = Math.abs(xs.getPixelForValue(7.5) - xs.getPixelForValue(0));
                      ctx.beginPath();
                      ctx.arc(cx, cy0, rHoop, 0, 2 * Math.PI);
                      ctx.stroke();
                      const x220L = xs.getPixelForValue(-220);
                      const x220R = xs.getPixelForValue(220);
                      ctx.beginPath();
                      ctx.moveTo(x220L, yB);
                      ctx.lineTo(x220L, zyT);
                      ctx.moveTo(x220R, yB);
                      ctx.lineTo(x220R, zyT);
                      ctx.stroke();
                      const r238 = Math.abs(xs.getPixelForValue(238) - xs.getPixelForValue(0));
                      ctx.beginPath();
                      ctx.arc(cx, cy0, r238, Math.PI * 0.73, Math.PI * 0.27, true);
                      ctx.stroke();
                      ctx.beginPath();
                      ctx.moveTo(xL, yT);
                      ctx.lineTo(xR, yT);
                      ctx.stroke();
                      ctx.restore();
                    },
                  };

                  const options = {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { labels: { color: "#cbd5e1" } },
                      tooltip: {
                        enabled: true,
                        callbacks: {
                          label(ctx: any) {
                            const x = ctx.parsed.x;
                            const y = ctx.parsed.y;
                            const datasetLabel = ctx.dataset.label;
                            const feet = Math.sqrt(x * x + y * y) / 10;
                            const meters = feet * 0.3048;
                            return `${datasetLabel} • x:${x.toFixed(0)}, y:${y.toFixed(0)} • dist: ${meters.toFixed(1)} m`;
                          },
                        },
                      },
                    },
                    scales: {
                      x: { type: "linear" as const, min: -250, max: 250, grid: { color: "rgba(30,58,138,0.2)" }, ticks: { color: "#94a3b8" } },
                      y: { type: "linear" as const, min: -50, max: 470, grid: { color: "rgba(30,58,138,0.2)" }, ticks: { color: "#94a3b8" } },
                    },
                  };

                  return <Scatter data={data} options={options} plugins={[courtPlugin]} />;
                })()}
              </div>
            </section>
          )}

          <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
            <h2 className="mb-3 text-xl font-semibold text-white">Comparativa</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-slate-800/70">
                  <tr>
                    <th className="px-3 py-2 text-left">Estadística</th>
                    <th className="px-3 py-2 text-right">Jugador</th>
                    <th className="px-3 py-2 text-right">Prom. Clúster</th>
                    <th className="px-3 py-2 text-right">Dif</th>
                    <th className="px-3 py-2 text-right">% Dif</th>
                  </tr>
                </thead>
                <tbody>
                  {data.comparison.map((c) => (
                    <tr key={c.stat} className="border-t border-blue-900/30">
                      <td className="px-3 py-2">{STAT_LABELS[c.stat] || c.stat}</td>
                      <td className="px-3 py-2 text-right">{c.player_value.toFixed(2)}</td>
                      <td className="px-3 py-2 text-right">{c.cluster_avg.toFixed(2)}</td>
                      <td className="px-3 py-2 text-right">{c.diff.toFixed(2)}</td>
                      <td className="px-3 py-2 text-right">{c.percent_diff.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">
            <h2 className="mb-3 text-xl font-semibold text-white">Áreas débiles</h2>
            {data.weak_areas.length === 0 ? (
              <p>No se detectaron áreas débiles destacadas.</p>
            ) : (
              <ul className="list-inside list-disc space-y-1">
                {data.weak_areas.map((w, i) => {
                  // Reemplaza el código de estadística por el nombre en español
                  const statCodeMatch = w.match(/^(\w+):/);
                  let label = null;
                  if (statCodeMatch) {
                    const code = statCodeMatch[1];
                    label = STAT_LABELS[code] || code;
                  }
                  // Reemplaza solo la primera ocurrencia, si statCodeMatch existe
                  const display = label && statCodeMatch && statCodeMatch[0]
                    ? w.replace(statCodeMatch[0], `${label}:`)
                    : w;
                  return <li key={i}>{display}</li>;
                })}
              </ul>
            )}
          </section>

          <div className="flex gap-3">
            <button onClick={onDownloadPdf} className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500">
              Descargar PDF
            </button>
            <Link href="/teams" className="self-center text-sky-400 hover:text-sky-300">
              Volver a equipos
            </Link>
          </div>
        </div>
      )}

      {!loading && !data && !error && (
        <div className="rounded-xl border border-blue-900/40 bg-slate-900/60 p-4 shadow-lg">Sin datos aún.</div>
      )}
    </div>
  );
}
