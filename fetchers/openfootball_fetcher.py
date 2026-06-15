import requests
import json
import pandas as pd
from pathlib import Path
from config import OPENFOOTBALL_URLS, MATCHES_CSV
from .base_fetcher import BaseFetcher

class OpenFootballFetcher(BaseFetcher):
    def __init__(self, data_dir: Path, logger):
        super().__init__("openfootball", data_dir, logger)

    def fetch(self) -> dict:
        try:
            all_matches = []

            for year, url in OPENFOOTBALL_URLS.items():
                self.rate_limit()

                resp = requests.get(url, timeout=10)
                resp.raise_for_status()

                data = resp.json()

                # Estructura: {"name": "...", "matches": [{...}, ...]}
                for match in data.get("matches", []):
                    parsed = self._parse_match(match, year)
                    if parsed:
                        all_matches.append(parsed)

            return {"success": True, "data": all_matches}

        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    def _parse_match(self, match: dict, year: int) -> dict | None:
        # Check for score in different formats
        # Format 1: score1/score2 fields
        score1 = match.get("score1")
        score2 = match.get("score2")

        # Format 2: score object with ft (full-time)
        if score1 is None or score2 is None:
            score_obj = match.get("score", {})
            if isinstance(score_obj, dict):
                ft = score_obj.get("ft", [])
                if len(ft) == 2:
                    score1, score2 = ft[0], ft[1]

        if score1 is None or score2 is None:
            return None

        team1 = match.get("team1", "").strip()
        team2 = match.get("team2", "").strip()
        date = match.get("date", "")

        if not team1 or not team2 or not date:
            return None

        return {
            "source": "openfootball",
            "match_id": f"of_{year}_{date}_{team1}_{team2}".replace(" ", "_"),
            "competition": "WC",
            "season": str(year),
            "date": date,
            "home_team": team1,
            "away_team": team2,
            "home_goals": int(score1),
            "away_goals": int(score2),
            "venue": match.get("ground", "unknown"),
        }

    def save(self, data: list) -> int:
        csv_path = MATCHES_CSV

        existing_ids = set()
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                existing = pd.read_csv(csv_path)
                existing_ids = set(existing["match_id"].tolist())
            except (pd.errors.EmptyDataError, KeyError):
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        new_rows = [d for d in data if d["match_id"] not in existing_ids]

        if new_rows:
            df_new = pd.DataFrame(new_rows)
            if existing.empty:
                df_all = df_new
            else:
                df_all = pd.concat([existing, df_new], ignore_index=True)
            df_all.to_csv(csv_path, index=False)

        return len(new_rows)