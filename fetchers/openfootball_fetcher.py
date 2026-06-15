import requests
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
import re
import logging
from difflib import SequenceMatcher

# Asumir que BaseFetcher está en fetchers/
from fetchers.base_fetcher import BaseFetcher


class OpenFootballFetcher(BaseFetcher):

    BASE_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master"
    WORLD_CUP_YEARS = [2010, 2014, 2018, 2022, 2026]
    MAX_RETRIES = 3
    TIMEOUT = 15

    def __init__(self, data_dir: Path, logger: logging.Logger):
        super().__init__("openfootball", data_dir, logger)
        self.min_interval = 2  # 2 segundos entre requests
        self.output_csv = data_dir / "openfootball_current.csv"

    def fetch(self) -> dict:
        all_matches = []
        errors = []

        for year in self.WORLD_CUP_YEARS:
            self.logger.info(f"[{self.name}] Fetching World Cup {year}...")

            try:
                year_matches = self._fetch_year(year)
                all_matches.extend(year_matches)
                self.logger.info(f"[{self.name}] ✓ {len(year_matches)} matches from {year}")

            except Exception as e:
                error_msg = f"Year {year}: {str(e)}"
                errors.append(error_msg)
                self.logger.warning(f"[{self.name}] ✗ {error_msg}")
                continue

        if not all_matches:
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
                "years": sorted(set(m["season"] for m in all_matches)),
                "errors": errors
            }
        }

    def _fetch_year(self, year: int) -> list:
        url = f"{self.BASE_URL}/{year}/worldcup.json"

        for attempt in range(self.MAX_RETRIES):
            try:
                self.rate_limit()

                resp = requests.get(
                    url,
                    headers={"User-Agent": "football-predictor/1.0"},
                    timeout=self.TIMEOUT
                )

                if resp.status_code == 404:
                    # 2026 puede no estar disponible aún
                    self.logger.debug(f"[{self.name}] {year} not available (404)")
                    return []

                resp.raise_for_status()

                data = resp.json()
                matches = self._parse_year_data(data, year)

                return matches

            except requests.exceptions.Timeout:
                self.logger.warning(f"[{self.name}] Timeout fetching {year}, attempt {attempt + 1}/{self.MAX_RETRIES}")
                if attempt < self.MAX_RETRIES - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"[{self.name}] Parse error for {year}: {e}")
                raise

        raise Exception(f"Failed to fetch {year} after {self.MAX_RETRIES} retries")

    def _parse_year_data(self, data: dict, year: int) -> list:
        matches = []
        match_list = []

        if "matches" in data:
            match_list = data["matches"]
        elif "rounds" in data:
            # Algunos años organizan por rondas
            for round_obj in data.get("rounds", []):
                match_list.extend(round_obj.get("matches", []))

        for match in match_list:
            try:
                parsed = self._parse_match(match, year)
                if parsed:
                    matches.append(parsed)
            except Exception as e:
                # Log pero continúa
                self.logger.debug(f"[{self.name}] Skipping malformed match: {e}")
                continue

        return matches

    def _parse_match(self, match: dict, year: int) -> Optional[dict]:
        team1 = self._normalize_team_name(match.get("team1", "").strip())
        team2 = self._normalize_team_name(match.get("team2", "").strip())

        if not team1 or not team2:
            return None

        date = match.get("date", "").strip()
        if not date or not self._is_valid_date(date):
            return None

        home_goals, away_goals = self._extract_score(match)

        if home_goals is None or away_goals is None:
            return None

        # Estadio (venue)
        venue = "unknown"
        if "stadium" in match:
            stadium = match["stadium"]
            if isinstance(stadium, dict):
                venue = stadium.get("name", "unknown")
            elif isinstance(stadium, str):
                venue = stadium
        elif "ground" in match:
            venue = match.get("ground", "unknown")

        # Grupo/Fase
        group = match.get("group", match.get("stage", "Unknown"))

        return {
            "source": "openfootball",
            "match_id": f"of_{year}_{date}_{team1}_{team2}".replace(" ", "_")[:80],
            "competition": "WC",
            "season": str(year),
            "date": date,
            "home_team": team1,
            "away_team": team2,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "venue": str(venue)[:100],
            "stage": str(group)[:50],
            "fetched_at": datetime.now().isoformat(),
        }

    def _extract_score(self, match: dict) -> tuple[Optional[int], Optional[int]]:
        if "score" in match:
            score = match["score"]
            if isinstance(score, dict):
                if "ft" in score:
                    ft = score["ft"]
                    if isinstance(ft, list) and len(ft) == 2:
                        return int(ft[0]), int(ft[1])
                elif "fulltime" in score:
                    ft = score["fulltime"]
                    if isinstance(ft, list) and len(ft) == 2:
                        return int(ft[0]), int(ft[1])

        if "score1" in match and "score2" in match:
            try:
                return int(match["score1"]), int(match["score2"])
            except (ValueError, TypeError):
                pass

        if "goals1" in match and "goals2" in match:
            try:
                return int(match["goals1"]), int(match["goals2"])
            except (ValueError, TypeError):
                pass

        if "ft1" in match and "ft2" in match:
            try:
                return int(match["ft1"]), int(match["ft2"])
            except (ValueError, TypeError):
                pass

        return None, None

    def _normalize_team_name(self, name: str) -> str:
        if not name:
            return ""

        name_map = {
            "united states": "USA",
            "us": "USA",
            "korea republic": "South Korea",
            "korea south": "South Korea",
            "korea dpr": "North Korea",
            "korea north": "North Korea",
            "ivory coast": "Côte d'Ivoire",
            "cote d'ivoire": "Côte d'Ivoire",
            "england": "England",
            "scotland": "Scotland",
            "wales": "Wales",
            "northern ireland": "Northern Ireland",
        }

        normalized = name_map.get(name.lower(), name)
        return normalized.title() if normalized else name

    def _is_valid_date(self, date_str: str) -> bool:
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        return bool(re.match(pattern, date_str))

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

        # Dedup por match_id (extra precaution)
        df_all = df_all.drop_duplicates(subset=["match_id"], keep="last")

        df_all.to_csv(self.output_csv, index=False)

        self.logger.info(f"[{self.name}] Saved {len(new_rows)} new records to {self.output_csv}")
        self.logger.info(f"[{self.name}] Total records: {len(df_all)}")

        return len(new_rows)