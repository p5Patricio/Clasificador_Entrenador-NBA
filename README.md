# Clasificador_Entrenador-NBA (Monorepo)

Este proyecto agrupa jugadores NBA por perfil estadístico (K-Means), permite analizar un jugador concreto contra el promedio de su clúster, y generar un reporte PDF con gráficos de radar y sugerencias de entrenamiento. Ahora está organizado como monorepo con `backend/` (FastAPI) y un futuro `frontend/` (Next.js).

## Estructura

- `backend/`
  - `api.py`: servidor FastAPI con endpoints REST.
  - `nba_data_processor.py`: preproceso, escalado, clustering y promedios por clúster.
  - `nba_player_analyzer.py`: utilidades de análisis, roles de clúster, gráficos y PDF.
  - `nba_player_data.py`: descarga y construye el dataset (Excel) desde `nba_api`.
  - `requirements.txt`: dependencias del backend.
- `frontend/` (por crear): proyecto Next.js 14 + Tailwind + shadcn/ui.
- `nba_active_player_stats_*.xlsx`: Excel del dataset (puede estar en la raíz; pásalo por ruta al inicializar el backend).

Nota: Se han dejado los archivos originales en la raíz solo de referencia; usa siempre los dentro de `backend/`.

## Requisitos

- Python 3.10+
- Node.js 18+ (para el frontend)
- Windows PowerShell (las instrucciones están pensadas para este shell)

## Backend: preparar entorno y ejecutar

```powershell
# 1) Crear y activar entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Instalar dependencias del backend
pip install -r .\backend\requirements.txt

# 3) Arrancar el servidor en desarrollo (desde la raíz del repo)
uvicorn backend.api:app --reload --port 8000
```

Inicializa el modelo (clustering) con un POST a `http://localhost:8000/cluster/init`:

```json
{
  "filepath": "nba_active_player_stats_2023-24_Regular_Season_100min.xlsx",
  "k": 5
}
```

El backend intentará resolver la ruta de forma relativa tanto a tu CWD como a la carpeta `backend/`. Si el archivo está en la raíz del repo, el valor del ejemplo funciona.

### Endpoints principales

- `GET /health`: estado del servidor.
- `POST /cluster/init`: carga el Excel, escala, hace K-Means y asigna roles.
- `GET /teams`: lista abreviaciones de equipos presentes en el dataset y cantidad de jugadores.
- `GET /players?team=LAL`: lista jugadores (opcional filtrar por equipo).
- `GET /player/{player_name}/analysis`: análisis JSON del jugador contra su clúster.
- `GET /player/{player_name}/report`: genera y devuelve un PDF con el reporte.

## Frontend: cómo crearlo en `frontend/`

Vamos a usar Next.js 14 con Tailwind CSS y shadcn/ui para un resultado moderno.

1) Crear el proyecto Next.js en la carpeta `frontend/`:

```powershell
# Desde la raíz del repo
npx create-next-app@latest frontend --typescript --eslint --tailwind --app --src-dir --import-alias "@/*"
```

2) Entrar en `frontend/` e instalar dependencias extra recomendadas:

```powershell
cd .\frontend
# Opcionales pero recomendados
npm install @radix-ui/react-icons class-variance-authority clsx tailwind-merge
```

3) Instalar shadcn/ui y generar componentes base:

```powershell
npx shadcn-ui@latest init
npx shadcn-ui@latest add button card input table select toast
```

4) Variables de entorno del frontend (si aplican): crea `.env.local` con la URL del backend si no es localhost:8000.

```powershell
echo "NEXT_PUBLIC_API_BASE=http://localhost:8000" > .env.local
```

5) Crear servicios de API (fetch) y páginas:
- Página Home: botón para `POST /cluster/init` (elige `k` y `filepath`).
- Página Equipos: `GET /teams` → seleccionar equipo → `GET /players?team=XXX`.
- Página Jugador: `GET /player/{name}/analysis` → mostrar radar (Chart.js) y tabla comparativa; botón para descargar PDF (`/player/{name}/report`).

6) Ejecutar el frontend en desarrollo:

```powershell
npm run dev
```

La web quedará en `http://localhost:3000` y consumirá el backend en `http://localhost:8000`.

## Troubleshooting

- Si `nba_api` falla, aumenta/reduce `time.sleep` en `backend/nba_player_data.py` y reintenta.
- Si no arranca `uvicorn`, verifica que el entorno virtual esté activado y que instalaste `backend/requirements.txt`.
- CORS: en producción, restringe `allow_origins` en `backend/api.py`.
