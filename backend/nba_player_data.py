import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
import time
import math

def get_nba_active_player_stats(season='2023-24', season_type='Regular Season', min_minutes_played=100):
    """
    Obtiene estadísticas de carrera por temporada para jugadores activos de la NBA
    y con un mínimo de minutos jugados en la temporada especificada.
    """
    active_nba_players = players.get_active_players()
    all_player_stats = []
    
    print(f"Obteniendo IDs de {len(active_nba_players)} jugadores activos...")
    
    numeric_stats_columns = [
        'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
        'BLK', 'TOV', 'PF', 'PTS'
    ]

    for i, player in enumerate(active_nba_players):
        player_id = player['id']
        player_name = player['full_name']
        
        time.sleep(0.5)
        
        print(f"Procesando jugador {i+1}/{len(active_nba_players)}: {player_name} (ID: {player_id})")

        try:
            career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
            player_df = pd.DataFrame()

            if season_type == 'Regular Season':
                player_df = career_stats.get_data_frames()[0]
            elif season_type == 'Playoffs':
                if len(career_stats.get_data_frames()) > 1:
                    player_df = career_stats.get_data_frames()[1]
                else:
                    continue
            else:
                print(f"Tipo de temporada '{season_type}' no manejado. Saltando a {player_name}.")
                continue

            season_stats = player_df[player_df['SEASON_ID'] == season].copy()
            
            if not season_stats.empty:
                for col in numeric_stats_columns:
                    if col in season_stats.columns:
                        season_stats[col] = pd.to_numeric(season_stats[col], errors='coerce')
                
                if 'MIN' in season_stats.columns and not math.isnan(season_stats['MIN'].iloc[0]) and season_stats['MIN'].iloc[0] >= min_minutes_played:
                    season_stats['PLAYER_NAME'] = player_name
                    all_player_stats.append(season_stats)

        except Exception as e:
            print(f"Error al obtener estadísticas para {player_name} (ID: {player_id}): {e}")
            
    if all_player_stats:
        combined_df = pd.concat(all_player_stats, ignore_index=True)
        return combined_df
    else:
        print(f"No se encontraron estadísticas para la temporada {season} ({season_type}) con {min_minutes_played} minutos mínimos.")
        return pd.DataFrame()

if __name__ == '__main__':
    target_season = '2023-24'
    target_season_type = 'Regular Season' 
    min_minutes = 100

    print(f"Iniciando la obtención de datos para la temporada {target_season} ({target_season_type}) para jugadores con al menos {min_minutes} minutos...")
    player_data = get_nba_active_player_stats(
        season=target_season, 
        season_type=target_season_type,
        min_minutes_played=min_minutes
    )

    if not player_data.empty:
        cols = ['PLAYER_NAME'] + [col for col in player_data.columns if col != 'PLAYER_NAME']
        player_data = player_data[cols]

        output_filename = f'nba_active_player_stats_{target_season}_{target_season_type.replace(" ", "_")}_{min_minutes}min.xlsx'
        player_data.to_excel(output_filename, index=False)
        print(f"\n¡Dataset guardado exitosamente en '{output_filename}'!")
        print(f"Dimensiones del dataset: {player_data.shape}")
        print("\nPrimeras 5 filas del dataset:")
        print(player_data.head())
    else:
        print("No se pudo obtener el dataset. Revisa la temporada, el tipo de temporada o el umbral de minutos.")
