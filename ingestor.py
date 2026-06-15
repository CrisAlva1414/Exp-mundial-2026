import time
import requests
import pandas as pd
from pathlib import Path
from config import API_KEY, BASE_URL, RATE_LIMIT_PER_MINUTE, DATA_PATH

class FootballIngestor:
    def __init__(self):
        self.headers = {"X-Auth-Token": API_KEY}
        self.min_interval = 60 / RATE_LIMIT_PER_MINUTE  # 6 segundos entre calls
        self.last_call = 0

    def _get(self, endpoint: str, params: dict = {}) -> dict:
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        url = f"{BASE_URL}/{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params)
        self.last_call = time.time()
        
        if resp.status_code == 429:
            print("Rate limit hit, esperando 60s...")
            time.sleep(60)
            return self._get(endpoint, params)  # retry
        
        resp.raise_for_status()
        return resp.json()

    def fetch_matches(self, competition_code: str, season: int = None) -> list[dict]:
        params = {"status": "FINISHED"}
        if season:
            params["season"] = season
        
        data = self._get(f"competitions/{competition_code}/matches", params)
        return data.get("matches", [])

    def parse_match(self, match: dict, competition: str) -> dict | None:
        score = match.get("score", {})
        ft = score.get("fullTime", {})
        
        if ft.get("home") is None:  # partido sin resultado aún
            return None
        
        return {
            "match_id":       match["id"],
            "competition":    competition,
            "season":         match.get("season", {}).get("startDate", "")[:4],
            "date":           match["utcDate"][:10],
            "home_team":      match["homeTeam"]["name"],
            "away_team":      match["awayTeam"]["name"],
            "home_goals":     ft["home"],
            "away_goals":     ft["away"],
            "venue":          match.get("venue", "unknown"),
            "stage":          match.get("stage", ""),
        }

    def ingest_competition(self, competition_code: str, seasons: list[int]) -> int:
        path = Path(DATA_PATH)
        path.parent.mkdir(exist_ok=True)
        
        # Cargar existentes para no duplicar
        if path.exists():
            existing = pd.read_csv(path)
            existing_ids = set(existing["match_id"].tolist())
        else:
            existing = pd.DataFrame()
            existing_ids = set()
        
        new_rows = []
        for season in seasons:
            print(f"  Fetching {competition_code} {season}...")
            matches = self.fetch_matches(competition_code, season)
            
            for match in matches:
                parsed = self.parse_match(match, competition_code)
                if parsed and parsed["match_id"] not in existing_ids:
                    new_rows.append(parsed)
                    existing_ids.add(parsed["match_id"])
        
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_all = pd.concat([existing, df_new], ignore_index=True)
            df_all.to_csv(path, index=False)
            print(f"  ✓ {len(new_rows)} partidos nuevos guardados")
        
        return len(new_rows)