from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from starlette.responses import FileResponse, Response

import pandas as pd
import numpy as np
import os
import datetime

# Importa utilidades existentes del paquete backend (importes relativos)
from .nba_data_processor import (
    load_and_preprocess_data,
    scale_data,
    perform_kmeans_clustering,
    analyze_clusters,
)
from .nba_player_analyzer import (
    assign_cluster_roles,
    generate_player_report_pdf,
    radar_stats_categories,
)
from .nba_player_data import get_nba_active_player_stats
from nba_api.stats.endpoints import commonplayerinfo, shotchartdetail
from nba_api.stats.static import players as static_players
import requests


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
    pass


class StatDelta(BaseModel):
    stat: str
    player_value: float
    cluster_avg: float
    diff: float
    percent_diff: float


class PlayerAnalysisResponse(BaseModel):
    player_name: str
    player_id: Optional[int] = None
    team: Optional[str]
    cluster_id: int
    cluster_role: str
    headshot_url: Optional[str] = None
    comparison: List[StatDelta]
    weak_areas: List[str]
    projected_stats: Dict[str, float]


class PlayerProfileResponse(BaseModel):
    player_id: int
    full_name: str
    team_id: Optional[int]
    team_abbreviation: Optional[str]
    team_name: Optional[str]
    height: Optional[str]
    weight: Optional[str]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    birthdate: Optional[str]
    age: Optional[float]
    headshot_url: Optional[str]


class PlayerShot(BaseModel):
    x: float
    y: float
    made: bool
    action_type: Optional[str] = None
    shot_zone_basic: Optional[str] = None
    shot_distance: Optional[float] = None


class PlayerShotsResponse(BaseModel):
    season: str
    season_type: str
    attempts: int
    makes: int
    shots: List[PlayerShot]


class RadarCategory(BaseModel):
    name: str
    stats: List[str]
    labels: List[str]
    series: Dict[str, List[float]]  # keys: player, cluster_avg, projected


class PlayerRadarsResponse(BaseModel):
    player_name: str
    cluster_id: int
    cluster_role: str
    categories: List[RadarCategory]


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
def _resolve_filepath(path: str) -> Optional[str]:
    """Intenta resolver la ruta probando varias ubicaciones relativas al paquete."""
    if not path:
        return None
    # Tal cual
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    # Relativo a este archivo y a su padre (raíz del repo)
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, '..'))
    cwd = os.getcwd()
    candidates = [
        os.path.join(here, path),
        os.path.join(repo_root, path),
        os.path.join(cwd, path),
        os.path.join(repo_root, 'data', path),
        os.path.join(here, 'data', path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


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

    cluster_id = int(row['CLUSTER'].iloc[0])
    team = row['TEAM_ABBREVIATION'].iloc[0] if 'TEAM_ABBREVIATION' in row.columns else None
    player_id = int(row['PLAYER_ID'].iloc[0]) if 'PLAYER_ID' in row.columns and not pd.isna(row['PLAYER_ID'].iloc[0]) else None
    role = CONTEXT.cluster_roles.get(cluster_id, "Rol Desconocido")

    player_stats_raw = row[stats_cols].iloc[0]
    cluster_avg_raw = CONTEXT.cluster_means.loc[cluster_id][stats_cols]

    comparison: List[StatDelta] = []
    weak_areas: List[str] = []
    projected_stats = player_stats_raw.copy()

    for stat in stats_cols:
        p_val = float(player_stats_raw.get(stat, np.nan))
        c_val = float(cluster_avg_raw.get(stat, np.nan))

        diff = p_val - c_val if (not np.isnan(p_val) and not np.isnan(c_val)) else np.nan
        percent = (diff / c_val * 100.0) if (not np.isnan(diff) and c_val not in (0.0, np.nan)) else 0.0

        comparison.append(StatDelta(
            stat=stat,
            player_value=float(0 if np.isnan(p_val) else p_val),
            cluster_avg=float(0 if np.isnan(c_val) else c_val),
            diff=float(0 if np.isnan(diff) else diff),
            percent_diff=float(percent),
        ))

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

    # Headshot URL vía CDN oficial de la NBA
    headshot_url = None
    if player_id is not None:
        # Resoluciones comunes: 260x190 o 1040x760
        headshot_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

    return PlayerAnalysisResponse(
        player_name=player_name,
        player_id=player_id,
        team=team,
        cluster_id=cluster_id,
        cluster_role=role,
        headshot_url=headshot_url,
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


@app.get("/")
def root():
    # Endpoint de cortesía para que no devuelva 404 en '/'
    return {
        "message": "NBA Player Analyzer backend running",
        "try": ["/health", "/docs", "/cluster/init"],
    }


@app.post("/cluster/init")
def init_cluster(req: InitRequest):
    resolved = _resolve_filepath(req.filepath)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {req.filepath}")
    try:
        CONTEXT.init_from_file(resolved, req.k, req.stats_columns)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e} [archivo='{req.filepath}', resuelto='{resolved}']")

    return {
        "message": "Modelo inicializado",
        "k": req.k,
        "players": int(CONTEXT.player_data_cleaned.shape[0]) if CONTEXT.player_data_cleaned is not None else 0,
        "clusters": int(len(CONTEXT.cluster_means)) if CONTEXT.cluster_means is not None else 0,
        "roles": CONTEXT.cluster_roles,
    }


class DataUpdateRequest(BaseModel):
    season: Optional[str] = None  # ejemplo '2024-25'. Si None, intentamos inferir la actual.
    season_type: str = "Regular Season"
    min_minutes: int = 100
    auto_init: bool = True


def _infer_current_season() -> str:
    # Regla simple: si estamos entre agosto (8) y diciembre (12), temporada X-(X+1)
    # si estamos entre enero (1) y julio (7), temporada (X-1)-X
    import datetime
    now = datetime.datetime.now()
    year = now.year
    if now.month >= 8:
        start = year
        end = (year + 1) % 100
    else:
        start = year - 1
        end = year % 100
    return f"{start}-{end:02d}"


@app.post("/data/update")
def update_dataset(req: DataUpdateRequest):
    season = req.season or _infer_current_season()
    try:
        df = get_nba_active_player_stats(season=season, season_type=req.season_type, min_minutes_played=req.min_minutes)
        if df is None or df.empty:
            raise HTTPException(status_code=500, detail=f"No se obtuvieron datos para {season} ({req.season_type})")
        cols = ['PLAYER_NAME'] + [c for c in df.columns if c != 'PLAYER_NAME']
        df = df[cols]
        filename = f"nba_active_player_stats_{season}_{req.season_type.replace(' ', '_')}_{req.min_minutes}min.xlsx"
        df.to_excel(filename, index=False)
        initialized = False
        if req.auto_init:
            resolved = _resolve_filepath(filename)
            CONTEXT.init_from_file(resolved, k=5, stats_columns=None)  # por defecto K=5
            initialized = True
        return {"message": "Dataset actualizado", "season": season, "season_type": req.season_type, "file": filename, "initialized": initialized}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando dataset: {e}")


def _get_player_id_by_name(name: str) -> Optional[int]:
    # Primero busca en el CONTEXT si está cargado
    if CONTEXT.player_data_cleaned is not None:
        df = CONTEXT.player_data_cleaned
        r = df[df['PLAYER_NAME'].str.lower() == name.lower()]
        if not r.empty and 'PLAYER_ID' in r.columns and not pd.isna(r['PLAYER_ID'].iloc[0]):
            return int(r['PLAYER_ID'].iloc[0])
    # Fallback a nba_api estático
    found = static_players.find_players_by_full_name(name)
    if found:
        return int(found[0]['id'])
    return None


def _compute_player_radars(player_name: str) -> PlayerRadarsResponse:
    if not CONTEXT.is_ready():
        raise HTTPException(status_code=400, detail="Inicializa el modelo con /cluster/init primero.")

    df = CONTEXT.player_data_cleaned
    stats_cols = [c for c in CONTEXT.stats_columns if c in df.columns]

    row = df[df['PLAYER_NAME'] == player_name]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Jugador '{player_name}' no encontrado en el dataset.")

    cluster_id = int(row['CLUSTER'].iloc[0])
    role = CONTEXT.cluster_roles.get(cluster_id, "Rol Desconocido")

    # Usa el mismo análisis para obtener proyección coherente
    analysis = _compute_player_analysis(player_name)

    player_stats_raw = row[stats_cols].iloc[0]
    cluster_avg_stats_raw = CONTEXT.cluster_means.loc[cluster_id][stats_cols]
    projected_stats_raw = pd.Series(analysis.projected_stats)

    # Global: todas las columnas de stats (excepto ids internos)
    exclude = {'PLAYER_ID', 'SEASON_ID', 'LEAGUE_ID', 'TEAM_ID', 'PLAYER_AGE'}
    all_stats = [s for s in stats_cols if s not in exclude]

    categories: List[RadarCategory] = []

    def build_category(name: str, stats_list: List[str]):
        current_stats = [s for s in stats_list if s in all_stats]
        if not current_stats:
            return
        p = player_stats_raw[current_stats].fillna(0)
        c = cluster_avg_stats_raw[current_stats].fillna(0)
        # projected puede no tener todos; align indices
        proj = projected_stats_raw.reindex(current_stats).fillna(0)
        # normalización por el máximo entre las tres series (como en PDF)
        max_val = max(p.max(), c.max(), proj.max())
        max_val = max_val if max_val and max_val > 0 else 1.0
        p_n = (p / max_val).tolist()
        c_n = (c / max_val).tolist()
        proj_n = (proj / max_val).tolist()
        # Etiquetas legibles: el frontend puede traducir, pero devolvemos ambas opciones
        from .nba_player_analyzer import STAT_NAMES_MAP
        labels = [STAT_NAMES_MAP.get(s, s) for s in current_stats]
        categories.append(RadarCategory(
            name=name,
            stats=current_stats,
            labels=labels,
            series={
                "player": p_n,
                "cluster_avg": c_n,
                "projected": proj_n,
            },
        ))

    # Global primero
    build_category("Global", all_stats)
    # Luego las categorías definidas para el PDF/UI
    for cat_name, stats_list in radar_stats_categories.items():
        build_category(cat_name, stats_list)

    return PlayerRadarsResponse(
        player_name=player_name,
        cluster_id=cluster_id,
        cluster_role=role,
        categories=categories,
    )


@app.get("/player/{player_name}/profile", response_model=PlayerProfileResponse)
def player_profile(player_name: str):
    pid = _get_player_id_by_name(player_name)
    if pid is None:
        raise HTTPException(status_code=404, detail=f"No se encontró el ID para '{player_name}'")
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_normalized_dict()
        data = info.get('CommonPlayerInfo', [{}])[0]
        height = data.get('HEIGHT')
        weight = data.get('WEIGHT')
        bdate = data.get('BIRTHDATE')
        age = None
        if bdate:
            try:
                # nba_api devuelve 'YYYY-MM-DDT00:00:00'
                dt = datetime.datetime.fromisoformat(bdate.replace('Z',''))
                age = (datetime.datetime.now() - dt).days / 365.25
            except Exception:
                age = None
        # Convertir alturas tipo 6-3 a cm si es el formato esperado
        height_cm = None
        if height and isinstance(height, str) and '-' in height:
            try:
                f, i = height.split('-')
                inches = int(f) * 12 + int(i)
                height_cm = round(inches * 2.54, 1)
            except Exception:
                height_cm = None
        weight_kg = None
        if weight:
            try:
                weight_kg = round(float(weight) * 0.45359237, 1)
            except Exception:
                weight_kg = None

        headshot_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
        return PlayerProfileResponse(
            player_id=pid,
            full_name=data.get('DISPLAY_FIRST_LAST') or player_name,
            team_id=data.get('TEAM_ID'),
            team_abbreviation=data.get('TEAM_ABBREVIATION'),
            team_name=data.get('TEAM_NAME'),
            height=height,
            weight=weight,
            height_cm=height_cm,
            weight_kg=weight_kg,
            birthdate=bdate,
            age=round(age, 1) if age else None,
            headshot_url=headshot_url,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo perfil: {e}")


@app.get("/player/{player_name}/shots", response_model=PlayerShotsResponse)
def player_shots(player_name: str, season: Optional[str] = None, season_type: str = Query(default="Regular Season")):
    pid = _get_player_id_by_name(player_name)
    if pid is None:
        raise HTTPException(status_code=404, detail=f"No se encontró el ID para '{player_name}'")
    season = season or _infer_current_season()
    try:
        sc = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=pid,
            season_nullable=season,
            season_type_all_star=season_type,
            context_measure_simple="FGA",
        ).get_data_frames()
        if not sc:
            raise HTTPException(status_code=500, detail="La API no devolvió datos de tiros")
        df = sc[0]
        shots: List[PlayerShot] = []
        makes = 0
        for _, row in df.iterrows():
            made = str(row.get('EVENT_TYPE', '')).lower().startswith('made')
            if made:
                makes += 1
            shots.append(PlayerShot(
                x=float(row.get('LOC_X', 0.0)),
                y=float(row.get('LOC_Y', 0.0)),
                made=made,
                action_type=row.get('ACTION_TYPE'),
                shot_zone_basic=row.get('SHOT_ZONE_BASIC'),
                shot_distance=float(row.get('SHOT_DISTANCE', 0.0)) if pd.notna(row.get('SHOT_DISTANCE')) else None,
            ))
        return PlayerShotsResponse(
            season=season,
            season_type=season_type,
            attempts=len(shots),
            makes=makes,
            shots=shots,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo tiros: {e}")


@app.get("/player/{player_name}/headshot")
def player_headshot(player_name: str):
    """Proxy opcional para servir la foto del jugador desde el backend (por si el CDN está bloqueado)."""
    pid = _get_player_id_by_name(player_name)
    if pid is None:
        raise HTTPException(status_code=404, detail=f"No se encontró el ID para '{player_name}'")
    url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise HTTPException(status_code=404, detail="Headshot no encontrado en CDN")
        return Response(content=r.content, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recuperando headshot: {e}")


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


@app.get("/player/{player_name}/radars", response_model=PlayerRadarsResponse)
def player_radars(player_name: str):
    """Devuelve datasets normalizados para radares (Global y por categorías),
    consistentes con la normalización usada en el PDF."""
    return _compute_player_radars(player_name)


@app.get("/player/{player_name}/report")
def player_report(player_name: str):
    if not CONTEXT.is_ready():
        raise HTTPException(status_code=400, detail="Inicializa el modelo con /cluster/init primero.")

    df = CONTEXT.player_data_cleaned
    row = df[df['PLAYER_NAME'] == player_name]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Jugador '{player_name}' no encontrado en el dataset.")

    cluster_id = int(row['CLUSTER'].iloc[0])
    player_stats_raw = row[CONTEXT.stats_columns].iloc[0]
    cluster_avg_stats_raw = CONTEXT.cluster_means.loc[cluster_id]

    analysis = _compute_player_analysis(player_name)

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

    detailed_drills_html = "<h3>Sugerencias de Entrenamiento (genéricas)</h3><ul>"
    for w in analysis.weak_areas:
        detailed_drills_html += f"<li>{w}</li>"
    detailed_drills_html += "</ul>"

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
