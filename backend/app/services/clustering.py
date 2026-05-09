from typing import Dict, Any
from fastapi import HTTPException
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from app.repositories.stats import StatsRepository


class ClusteringService:
    def __init__(self, stats_repo: StatsRepository):
        self.stats_repo = stats_repo

    def init_clusters(self, season_id: int, k: int = 5) -> Dict[str, Any]:
        stats = self.stats_repo.find_by_season(season_id)
        if not stats:
            raise HTTPException(status_code=404, detail=f"No player stats found for season {season_id}")

        feature_cols = [
            "min", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
            "ftm", "fta", "ft_pct", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts",
        ]

        data = []
        stat_ids = []
        for s in stats:
            row = [getattr(s, col, 0) or 0 for col in feature_cols]
            data.append(row)
            stat_ids.append(s.id)

        df = pd.DataFrame(data, columns=feature_cols)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled)

        assignments = {stat_ids[i]: int(clusters[i]) for i in range(len(stat_ids))}
        self.stats_repo.update_clusters(season_id, assignments)

        df["cluster"] = clusters
        cluster_means = df.groupby("cluster")[feature_cols].mean()
        roles = self._assign_roles(cluster_means)

        return {
            "season_id": season_id,
            "k": k,
            "players": len(stats),
            "clusters": k,
            "roles": roles,
        }

    def _assign_roles(self, cluster_means: pd.DataFrame) -> Dict[int, str]:
        roles = {}
        for cluster_id, stats in cluster_means.iterrows():
            role = "Jugador de Rol General"

            if stats["min"] < 200 and stats["pts"] < 100:
                role = "Jugador de Limite de Roster / Desarrollo"
            elif stats["blk"] > 70 and stats["reb"] > 300 and stats["min"] > 800:
                role = "Protector de Aro / Pivot Defensivo"
            elif stats["fg_pct"] > 0.60 and stats["reb"] > 400 and stats["pts"] > 300:
                role = "Centro Dominante / Interior Eficiente"
            elif stats["reb"] > 350 and stats["min"] > 800 and stats["pts"] < 600:
                role = "Hombre Grande Rebotador / De Rol"
            elif stats["ast"] > 350 and stats["min"] > 1200:
                if stats["pts"] > 800 and stats["fg3m"] > 80:
                    role = "Base Creador de Juego / Anotador"
                else:
                    role = "Base Creador de Juego Puro"
            elif stats["ast"] > 150 and stats["min"] > 500 and stats["pts"] < 500:
                if stats["fg3m"] > 70 and stats["fg3_pct"] > 0.35:
                    role = "Base de Rol / Facilitador (Tirador)"
                else:
                    role = "Base de Rol / Manejador de Balon"
            elif stats["fg3_pct"] > 0.38 and stats["fg3m"] > 150 and stats["fga"] > 600:
                role = "Especialista en Triples / Escolta Anotador"
            elif stats["pts"] > 800 and stats["fga"] > 650:
                role = "Anotador de Volumen / Alero Ofensivo"
            elif (stats["stl"] > 80 or stats["blk"] > 40) and stats["min"] > 700:
                role = "Defensor de Rol"
            elif stats["pts"] > 500 and stats["ast"] > 150 and stats["reb"] > 250 and stats["min"] > 1000 and stats["fg_pct"] > 0.45:
                role = "Jugador Completo (All-Around)"

            roles[int(cluster_id)] = role
        return roles
