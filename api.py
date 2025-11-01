from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from starlette.responses import FileResponse

import pandas as pd
import numpy as np
import os

# Importa utilidades existentes del repo
from nba_data_processor import (
    load_and_preprocess_data,
    scale_data,
    perform_kmeans_clustering,
    analyze_clusters,
)
from nba_player_analyzer import (
    assign_cluster_roles,
    generate_player_report_pdf,
    radar_stats_categories,
    STAT_NAMES_MAP,
)

# -------------------------------
# Configuración de la aplicación
# -------------------------------
app = FastAPI(title="NBA Player Analyzer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Modelos Pydantic (I/O)
# -------------------------------
class InitRequest(BaseModel):
    filepath: str = "nba_active_player_stats_2023-24_Regular_Season_100min.xlsx"
    k: int = 5
    stats_columns: Optional[List[str]] = None

class Team(BaseModel):
    abbreviation: str
    players: int

class PlayerAnalysisRequest(BaseModel):
    # Reservado por si luego queremos parámetros extra (p.e., threshold)
    pass

class StatDelta(BaseModel):
    stat: str
    player_value: float
    cluster_avg: float
    diff: float
    percent_diff: float

class PlayerAnalysisResponse(BaseModel):
    player_name: str
    team: Optional[str]
    cluster_id: int
    cluster_role: str
    comparison: List[StatDelta]
    weak_areas: List[str]
    projected_stats: Dict[str, float]

# -------------------------------
# Contexto en memoria (estado)
# -------------------------------
class DataContext:
    def __init__(self):
        self.filepath: Optional[str] = None
        self.stats_columns: Optional[List[str]] = None
        self.player_data_cleaned: Optional[pd.DataFrame] = None
        self.scaled_stats_df: Optional[pd.DataFrame] = None
        self.scaler = None
        self.kmeans_model = None
        self.clusters: Optional[np.ndarray] = None
        self.cluster_means: Optional[pd.DataFrame] = None
        self.cluster_roles: Optional[Dict[int, str]] = None

    def is_ready(self) -> bool:
        return (
            self.player_data_cleaned is not None
            and self.cluster_means is not None
            and self.cluster_roles is not None
            and self.kmeans_model is not None
        )

    def init_from_file(self, filepath: str, k: int, stats_columns: Optional[List[str]] = None):
        default_stats = [
            'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
            'BLK', 'TOV', 'PF', 'PTS', 'GP', 'GS'
        ]
        self.filepath = filepath
        self.stats_columns = stats_columns or default_stats

        stats_for_clustering, player_data_cleaned, _ = load_and_preprocess_data(filepath, self.stats_columns)
        if stats_for_clustering is None:
            raise ValueError("No se pudo cargar/preprocesar el dataset.")

        scaled_stats_df, scaler = scale_data(stats_for_clustering)
        clusters, kmeans_model = perform_kmeans_clustering(scaled_stats_df, k)

        player_data_cleaned = player_data_cleaned.copy()
        player_data_cleaned['CLUSTER'] = clusters

        cluster_means = analyze_clusters(player_data_cleaned, self.stats_columns)
        cluster_roles = assign_cluster_roles(cluster_means)

        # Persistir en el contexto
        self.player_data_cleaned = player_data_cleaned
        self.scaled_stats_df = scaled_stats_df
        self.scaler = scaler
        self.kmeans_model = kmeans_model
        self.clusters = clusters
        self.cluster_means = cluster_means
        self.cluster_roles = cluster_roles

CONTEXT = DataContext()

# -------------------------------
# Utilidades internas
# -------------------------------

def _get_teams_from_dataset(df: pd.DataFrame) -> List[Team]:
    counts = (
        df[df['TEAM_ABBREVIATION'].notna()]
        .groupby('TEAM_ABBREVIATION')
        .size()
        .sort_values(ascending=False)
    )
    return [Team(abbreviation=abbr, players=int(count)) for abbr, count in counts.items()]


def _compute_player_analysis(player_name: str) -> PlayerAnalysisResponse:
    if not CONTEXT.is_ready():
        raise HTTPException(status_code=400, detail="El modelo aún no está inicializado. Llama a /cluster/init primero.")

    df = CONTEXT.player_data_cleaned
    stats_cols = [c for c in CONTEXT.stats_columns if c in df.columns]

    row = df[df['PLAYER_NAME'] == player_name]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Jugador '{player_name}' no encontrado en el dataset.")

    # Extraer datos base
    cluster_id = int(row['CLUSTER'].iloc[0])
    team = row['TEAM_ABBREVIATION'].iloc[0] if 'TEAM_ABBREVIATION' in row.columns else None
    role = CONTEXT.cluster_roles.get(cluster_id, "Rol Desconocido")

    player_stats_raw = row[stats_cols].iloc[0]
    cluster_avg_raw = CONTEXT.cluster_means.loc[cluster_id][stats_cols]

    # Comparación y deltas
    comparison: List[StatDelta] = []
    weak_areas: List[str] = []

    projected_stats = player_stats_raw.copy()

    for stat in stats_cols:
        p_val = float(player_stats_raw.get(stat, np.nan))
        c_val = float(cluster_avg_raw.get(stat, np.nan))

        # Dif y %
        diff = p_val - c_val if (not np.isnan(p_val) and not np.isnan(c_val)) else np.nan
        percent = (diff / c_val * 100.0) if (not np.isnan(diff) and c_val not in (0.0, np.nan)) else 0.0

        comparison.append(StatDelta(
            stat=stat,
            player_value=float(0 if np.isnan(p_val) else p_val),
            cluster_avg=float(0 if np.isnan(c_val) else c_val),
            diff=float(0 if np.isnan(diff) else diff),
            percent_diff=float(percent),
        ))

        # Lógica de áreas débiles + proyección (coherente con tu script)
        if np.isnan(p_val) or np.isnan(c_val):
            continue
        if stat.endswith('_PCT'):
            gp = float(row['GP'].iloc[0]) if 'GP' in row.columns and not pd.isna(row['GP'].iloc[0]) else 0
            if gp > 10 and (c_val - p_val) > 0.05:
                weak_areas.append(f"{stat}: {p_val:.3f} (Promedio Clúster: {c_val:.3f}) - [Necesita mejorar puntería/eficiencia]")
                projected_stats[stat] = min(p_val + 0.03, c_val + 0.01)
        elif stat in ['TOV', 'PF']:
            if c_val > 0 and p_val > (c_val * 1.20):
                weak_areas.append(f"{stat}: {p_val:.2f} (Promedio Clúster: {c_val:.2f}) - [Reducir {stat}]")
                projected_stats[stat] = max(p_val * 0.90, c_val * 0.95)
        else:
            if c_val > 0 and p_val < (c_val * 0.75):
                weak_areas.append(f"{stat}: {p_val:.2f} (Promedio Clúster: {c_val:.2f}) - [Necesita mejorar {stat}]")
                projected_stats[stat] = min(p_val * 1.15, c_val * 0.95)

    return PlayerAnalysisResponse(
        player_name=player_name,
        team=team,
        cluster_id=cluster_id,
        cluster_role=role,
        comparison=comparison,
        weak_areas=weak_areas,
        projected_stats={s: float(projected_stats[s]) for s in stats_cols if not pd.isna(projected_stats[s])},
    )

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "initialized": CONTEXT.is_ready()}


@app.post("/cluster/init")
def init_cluster(req: InitRequest):
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {req.filepath}")
    try:
        CONTEXT.init_from_file(req.filepath, req.k, req.stats_columns)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": "Modelo inicializado",
        "k": req.k,
        "players": int(CONTEXT.player_data_cleaned.shape[0]) if CONTEXT.player_data_cleaned is not None else 0,
        "clusters": int(len(CONTEXT.cluster_means)) if CONTEXT.cluster_means is not None else 0,
        "roles": CONTEXT.cluster_roles,
    }


@app.get("/teams", response_model=List[Team])
def get_teams():
    if not CONTEXT.is_ready():
        raise HTTPException(status_code=400, detail="Inicializa el modelo con /cluster/init primero.")
    return _get_teams_from_dataset(CONTEXT.player_data_cleaned)


@app.get("/players", response_model=List[str])
def get_players(team: Optional[str] = Query(default=None, description="Abreviación de equipo, ej. LAL")):
    if not CONTEXT.is_ready():
        raise HTTPException(status_code=400, detail="Inicializa el modelo con /cluster/init primero.")
    df = CONTEXT.player_data_cleaned
    q = df
    if team:
        q = q[q['TEAM_ABBREVIATION'] == team]
    return sorted(q['PLAYER_NAME'].dropna().unique().tolist())


@app.get("/player/{player_name}/analysis", response_model=PlayerAnalysisResponse)
def analyze_player(player_name: str):
    return _compute_player_analysis(player_name)


@app.get("/player/{player_name}/report")
def player_report(player_name: str):
    if not CONTEXT.is_ready():
        raise HTTPException(status_code=400, detail="Inicializa el modelo con /cluster/init primero.")

    # Reconstruir insumos para la función de PDF
    df = CONTEXT.player_data_cleaned
    row = df[df['PLAYER_NAME'] == player_name]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Jugador '{player_name}' no encontrado en el dataset.")

    cluster_id = int(row['CLUSTER'].iloc[0])
    player_stats_raw = row[CONTEXT.stats_columns].iloc[0]
    cluster_avg_stats_raw = CONTEXT.cluster_means.loc[cluster_id]

    # Reusar la lógica de análisis para proyectados y weak areas
    analysis = _compute_player_analysis(player_name)

    # Construir comparison_df similar al script original
    comp_rows = []
    for c in analysis.comparison:
        comp_rows.append({
            'stat': c.stat,
            'Player Stats': c.player_value,
            'Cluster Average': c.cluster_avg,
            'Difference': c.diff,
            'Percentage Difference': c.percent_diff,
        })
    comparison_df = pd.DataFrame(comp_rows).set_index('stat')

    # detailed_drills_html: por simplicidad, lista de áreas débiles como bullets
    detailed_drills_html = "<h3>Sugerencias de Entrenamiento (genéricas)</h3><ul>"
    for w in analysis.weak_areas:
        detailed_drills_html += f"<li>{w}</li>"
    detailed_drills_html += "</ul>"

    # Generar PDF
    try:
        generate_player_report_pdf(
            player_name=player_name,
            player_row=row,
            player_cluster=cluster_id,
            cluster_roles=CONTEXT.cluster_roles,
            player_stats_raw=player_stats_raw,
            cluster_avg_stats_raw=cluster_avg_stats_raw,
            projected_stats_raw=pd.Series(analysis.projected_stats),
            comparison_df=comparison_df,
            weak_areas_list=analysis.weak_areas,
            detailed_drills_html=detailed_drills_html,
            all_stats_columns=CONTEXT.stats_columns,
            radar_categories=radar_stats_categories,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar el PDF: {e}")

    pdf_path = f"Reporte_{player_name.replace(' ', '_')}.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="No se encontró el PDF generado.")
    return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))
