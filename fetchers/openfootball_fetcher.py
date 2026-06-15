import requests
import json
import pandas as pd
from pathlib import Path
from config import OPENFOOTBALL_REPO
from .base_fetcher import BaseFetcher

class OpenFootballFetcher(BaseFetcher):
    def __init__(self, data_dir: Path, logger):
        super().__init__("openfootball", data_dir, logger)
    
    def fetch(self) -> dict:
        try:
            all_matches = []
            
            for year in [2014, 2018, 2022, 2026]:
                self.rate_limit()
                
                url = f"{OPENFOOTBALL_REPO}/{year}/worldcup.json"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                
                data = resp.json()
                
                # Estructura: {"rounds": [{"matches": [...]}]}
                for round_data in data.get("rounds", []):
                    for match in round_data.get("matches", []):
                        parsed = self._parse_match(match, year)
                        if parsed:
                            all_matches.append(parsed)
            
            return {"success": True, "data": all_matches}
        
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}
    
    def _parse_match(self, match: dict, year: int) -> dict | None:
        result = match.get("result")
        if result is None:
            return None
        
        return {
            "source": "openfootball",
            "match_id": f"of_{year}_{match.get('id', match['date'])}",
            "competition": "WC",
            "season": str(year),
            "date": match["date"],
            "home_team": match["team1"],
            "away_team": match["team2"],
            "home_goals": result,
            "away_goals": match.get("result2", result),  # Formato específico de OF
        }
    
    def save(self, data: list) -> int:
        csv_path = self.data_dir / "wc_historical.csv"
        
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