import pandas as pd
from pathlib import Path
from config import KAGGLE_RESULTS_CSV, MATCHES_CSV
from .base_fetcher import BaseFetcher


class KaggleFetcher(BaseFetcher):
    def __init__(self, data_dir: Path, logger):
        super().__init__("kaggle", data_dir, logger)

    def fetch(self) -> dict:
        """Read Kaggle dataset if it exists"""
        if not KAGGLE_RESULTS_CSV.exists():
            self.logger.info(
                f"[{self.name}] Kaggle CSV not found at {KAGGLE_RESULTS_CSV}\n"
                f"  Download manually from: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017\n"
                f"  Place results.csv in data/raw/ directory"
            )
            return {"success": False, "error": f"File not found: {KAGGLE_RESULTS_CSV}", "data": []}

        try:
            df = pd.read_csv(KAGGLE_RESULTS_CSV)
            records = self._normalize_dataframe(df)
            return {"success": True, "data": records}
        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    def _normalize_dataframe(self, df: pd.DataFrame) -> list:
        """Normalize Kaggle dataset to standard schema"""
        records = []
        for _, row in df.iterrows():
            try:
                date = str(row.get("date", "")).strip()
                home_team = str(row.get("home_team", "")).strip()
                away_team = str(row.get("away_team", "")).strip()
                home_score = int(row.get("home_score", 0))
                away_score = int(row.get("away_score", 0))
                tournament = str(row.get("tournament", "")).strip()
                city = str(row.get("city", "unknown")).strip()

                if not date or not home_team or not away_team:
                    continue

                record = {
                    "source": "kaggle",
                    "match_id": f"kg_{date}_{home_team}_{away_team}".replace(" ", "_"),
                    "competition": tournament if tournament else "OTHER",
                    "season": date[:4],
                    "date": date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_score,
                    "away_goals": away_score,
                    "venue": city,
                }
                records.append(record)
            except (ValueError, KeyError):
                continue

        return records

    def save(self, data: list) -> int:
        """Merge with existing matches.csv, deduplicating by match_id"""
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
            df_all = df_all.sort_values("date").reset_index(drop=True)
            df_all.to_csv(csv_path, index=False)

        return len(new_rows)
