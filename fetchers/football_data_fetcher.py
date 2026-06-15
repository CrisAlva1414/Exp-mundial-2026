import requests
import json
import time
import pandas as pd
from pathlib import Path
from config import FD_API_KEY, FD_BASE_URL, FD_RATE_LIMIT_PER_MINUTE, COMPETITIONS, STATE_FILE, MATCHES_CSV
from .base_fetcher import BaseFetcher

class FootballDataFetcher(BaseFetcher):
    def __init__(self, data_dir: Path, logger):
        super().__init__("football_data", data_dir, logger)
        self.headers = {"X-Auth-Token": FD_API_KEY}
        self.min_interval = 60 / FD_RATE_LIMIT_PER_MINUTE  # ~6 seg
        self.base_url = FD_BASE_URL
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def fetch(self) -> dict:
        all_matches = []

        if "football_data" not in self.state:
            self.state["football_data"] = {}

        for comp_code, comp_info in COMPETITIONS.items():
            try:
                for season in comp_info["seasons"]:
                    state_key = f"{comp_code}_{season}"

                    # Check if already done
                    if self.state["football_data"].get(state_key) == "done":
                        self.logger.info(f"[{self.name}] Skipping {comp_code}/{season} (already completed)")
                        continue

                    self.rate_limit()

                    url = f"{self.base_url}/competitions/{comp_code}/matches"
                    params = {"season": season, "status": "FINISHED"}

                    resp = requests.get(url, headers=self.headers, params=params, timeout=10)

                    if resp.status_code == 429:
                        self.logger.warning("Rate limit hit, waiting 60s...")
                        time.sleep(60)
                        resp = requests.get(url, headers=self.headers, params=params)

                    resp.raise_for_status()
                    data = resp.json()

                    for match in data.get("matches", []):
                        parsed = self._parse_match(match, comp_code)
                        if parsed:
                            all_matches.append(parsed)

                    # Mark as done after successful fetch
                    self.state["football_data"][state_key] = "done"
                    self._save_state()

            except Exception as e:
                self.logger.error(f"Error fetching {comp_code}: {e}")
                continue

        return {"success": True, "data": all_matches}

    def _parse_match(self, match: dict, comp_code: str) -> dict | None:
        score = match.get("score", {})
        ft = score.get("fullTime", {})

        if ft.get("home") is None:
            return None

        return {
            "source": "football_data",
            "match_id": f"fd_{match['id']}",
            "competition": comp_code,
            "season": match.get("season", {}).get("startDate", "")[:4],
            "date": match["utcDate"][:10],
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"],
            "home_goals": ft["home"],
            "away_goals": ft["away"],
            "venue": match.get("venue", "unknown"),
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