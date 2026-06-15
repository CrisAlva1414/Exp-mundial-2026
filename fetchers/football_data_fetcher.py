import requests
import json
import time
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fetchers.base_fetcher import BaseFetcher


class FootballDataFetcher(BaseFetcher):

    BASE_URL = "https://api.football-data.org/v4"
    RATE_LIMIT_PER_MINUTE = 10
    MIN_INTERVAL = 60 / RATE_LIMIT_PER_MINUTE  # ~6 segundos

    COMPETITIONS = {
        "WC": {"name": "FIFA World Cup", "seasons": [2022]},
        "CL": {"name": "UEFA Champions League", "seasons": [2023, 2024]},
        "BL1": {"name": "Bundesliga", "seasons": [2023, 2024]},
        "DED": {"name": "Eredivisie", "seasons": [2023, 2024]},
        "BSA": {"name": "Brasileirao", "seasons": [2023, 2024]},
        "PD": {"name": "La Liga", "seasons": [2023, 2024]},
        "FL1": {"name": "Ligue 1", "seasons": [2023, 2024]},
        "ELC": {"name": "Championship", "seasons": [2023, 2024]},
        "PPL": {"name": "Primeira Liga", "seasons": [2023, 2024]},
        "EC": {"name": "European Championship", "seasons": [2024]},
        "SA": {"name": "Serie A", "seasons": [2023, 2024]},
        "PL": {"name": "Premier League", "seasons": [2023, 2024]},
    }

    def __init__(
        self,
        data_dir: Path,
        logger: logging.Logger,
        api_key: Optional[str] = "hola",
        test_mode: bool = False
    ):

        super().__init__("football_data", data_dir, logger)
        self.api_key = api_key
        self.test_mode = test_mode
        self.min_interval = self.MIN_INTERVAL
        self.output_csv = data_dir / "datafootball_current.csv"
        self.state_file = data_dir / "state_football_data.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"[{self.name}] Could not load state: {e}")
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def fetch(self) -> dict:
        if self.test_mode:
            self.logger.info(f"[{self.name}] TEST MODE - skipping API calls")
            return {
                "success": True,
                "data": [],
                "metadata": {"test_mode": True}
            }

        if not self.api_key:
            self.logger.warning(f"[{self.name}] No API key provided - running in limited mode")
            return {
                "success": False,
                "error": "No API key provided. Set FD_API_KEY environment variable or pass api_key parameter.",
                "data": []
            }

        all_matches = []
        errors = []

        if "football_data" not in self.state:
            self.state["football_data"] = {}

        for comp_code, comp_info in self.COMPETITIONS.items():
            for season in comp_info["seasons"]:
                state_key = f"{comp_code}_{season}"

                # Saltar si ya fue descargado
                if self.state["football_data"].get(state_key) == "done":
                    self.logger.debug(f"[{self.name}] Skipping {comp_code}/{season} (already completed)")
                    continue

                try:
                    self.logger.info(f"[{self.name}] Fetching {comp_code}/{season}...")
                    year_matches = self._fetch_competition(comp_code, season)
                    all_matches.extend(year_matches)

                    # Marcar como completado
                    self.state["football_data"][state_key] = "done"
                    self._save_state()

                    self.logger.info(f"[{self.name}] ✓ {len(year_matches)} matches")

                except requests.exceptions.HTTPError as e:
                    error_msg = f"{comp_code}/{season}: HTTP {e.response.status_code}"
                    errors.append(error_msg)
                    self.logger.warning(f"[{self.name}] ✗ {error_msg}")
                    continue

                except Exception as e:
                    error_msg = f"{comp_code}/{season}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.warning(f"[{self.name}] ✗ {error_msg}")
                    continue

        if not all_matches and errors:
            return {
                "success": False,
                "error": f"No matches downloaded. Errors: {'; '.join(errors)}",
                "data": []
            }

        return {
            "success": True,
            "data": all_matches,
            "metadata": {
                "total_matches": len(all_matches),
                "competitions": sorted(set(m["competition"] for m in all_matches)),
                "errors": errors
            }
        }

    def _fetch_competition(self, comp_code: str, season: int) -> list:
        url = f"{self.BASE_URL}/competitions/{comp_code}/matches"
        params = {
            "season": season,
            "status": "FINISHED"
        }

        headers = {
            "X-Auth-Token": self.api_key,
            "User-Agent": "football-predictor/1.0"
        }

        self.rate_limit()

        resp = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=15
        )

        # Manejo de rate limit
        if resp.status_code == 429:
            self.logger.warning(f"[{self.name}] Rate limit hit (429). Waiting 60s...")
            time.sleep(60)
            self.rate_limit()
            resp = requests.get(url, headers=headers, params=params, timeout=15)

        if resp.status_code != 200:
            self.logger.error(
                f"[{self.name}] HTTP {resp.status_code}: {resp.text}"
            )

        resp.raise_for_status()

        data = resp.json()
        matches = []

        for match in data.get("matches", []):
            try:
                parsed = self._parse_match(match, comp_code)
                if parsed:
                    matches.append(parsed)
            except Exception as e:
                self.logger.debug(f"[{self.name}] Skipping malformed match: {e}")
                continue

        return matches

    def _parse_match(self, match: dict, comp_code: str) -> Optional[dict]:
        score = match.get("score", {})
        ft = score.get("fullTime", {})

        home_goals = ft.get("home")
        away_goals = ft.get("away")

        if home_goals is None or away_goals is None:
            return None

        home_team = match.get("homeTeam", {}).get("name", "").strip()
        away_team = match.get("awayTeam", {}).get("name", "").strip()

        if not home_team or not away_team:
            return None

        date = match.get("utcDate", "")[:10]
        if not date or not self._is_valid_date(date):
            return None

        return {
            "source": "football_data",
            "match_id": f"fd_{match.get('id', '')}",
            "competition": comp_code,
            "season": str(match.get("season", {}).get("startDate", "")[:4]),
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "venue": match.get("venue", "unknown")[:100],
            "fetched_at": datetime.now().isoformat(),
        }

    def _is_valid_date(self, date_str: str) -> bool:
        try:
            parts = date_str.split('-')
            if len(parts) != 3:
                return False
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31
        except (ValueError, IndexError):
            return False

    def save(self, data: list) -> int:
        if not data:
            return 0

        # Leer existing si existe
        existing_ids = set()
        if self.output_csv.exists() and self.output_csv.stat().st_size > 0:
            try:
                existing_df = pd.read_csv(self.output_csv)
                existing_ids = set(existing_df["match_id"].tolist())
                self.logger.info(f"[{self.name}] Found {len(existing_ids)} existing records")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Could not read existing CSV: {e}")
                existing_df = pd.DataFrame()
        else:
            existing_df = pd.DataFrame()

        # Filtrar nuevos
        new_rows = [d for d in data if d["match_id"] not in existing_ids]

        if not new_rows:
            self.logger.info(f"[{self.name}] No new records to save")
            return 0

        # Guardar
        df_new = pd.DataFrame(new_rows)

        if existing_df.empty:
            df_all = df_new
        else:
            df_all = pd.concat([existing_df, df_new], ignore_index=True)

        # Sort by date
        df_all = df_all.sort_values("date").reset_index(drop=True)

        # Dedup
        df_all = df_all.drop_duplicates(subset=["match_id"], keep="last")

        df_all.to_csv(self.output_csv, index=False)

        self.logger.info(f"[{self.name}] Saved {len(new_rows)} new records to {self.output_csv}")
        self.logger.info(f"[{self.name}] Total records: {len(df_all)}")

        return len(new_rows)