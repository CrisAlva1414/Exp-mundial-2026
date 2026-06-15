import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from config import ELORATINGS_CSV_URL
from .base_fetcher import BaseFetcher

class EloratingsFetcher(BaseFetcher):
    def __init__(self, data_dir: Path, logger):
        super().__init__("eloratings", data_dir, logger)
    
    def fetch(self) -> dict:
        try:
            self.rate_limit()
            resp = requests.get(ELORATINGS_CSV_URL, timeout=15)
            resp.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(StringIO(resp.text))
            
            # Normalizar columnas
            df_clean = df.rename(columns={
                "Date": "date",
                "Home Team": "home_team",
                "Away Team": "away_team",
                "Home Goals": "home_goals",
                "Away Goals": "away_goals",
                "Home Elo": "home_elo",
                "Away Elo": "away_elo",
                "Home Elo Change": "home_elo_change",
                "Away Elo Change": "away_elo_change",
            })
            
            df_clean["source"] = "eloratings"
            df_clean["match_id"] = df_clean.apply(
                lambda row: f"elo_{row['date']}_{row['home_team']}_{row['away_team']}", 
                axis=1
            )
            
            records = df_clean.to_dict("records")
            return {"success": True, "data": records}
        
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}
    
    def save(self, data: list) -> int:
        csv_path = self.data_dir / "elo_ratings.csv"
        
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            existing_ids = set(existing["match_id"].tolist())
        else:
            existing = pd.DataFrame()
            existing_ids = set()
        
        new_rows = [d for d in data if d["match_id"] not in existing_ids]
        
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_all = pd.concat([existing, df_new], ignore_index=True)
            df_all.to_csv(csv_path, index=False)
        
        return len(new_rows)