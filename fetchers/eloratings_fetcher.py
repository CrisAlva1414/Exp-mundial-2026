import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from config import ELORATINGS_URL, ELO_CSV
from .base_fetcher import BaseFetcher

class EloratingsFetcher(BaseFetcher):
    def __init__(self, data_dir: Path, logger):
        super().__init__("eloratings", data_dir, logger)

    def fetch(self) -> dict:
        """
        Fetch ELO ratings from eloratings.net.
        Note: The site uses JavaScript rendering, so direct scraping won't work.
        """
        try:
            self.rate_limit()
            headers = {"User-Agent": "Mozilla/5.0 (compatible; football-predictor/1.0)"}
            resp = requests.get(ELORATINGS_URL, headers=headers, timeout=15)
            resp.raise_for_status()

            # Check if there's a CSV export or API endpoint
            # Try direct CSV export if available
            csv_url = "https://www.eloratings.net/export/all.csv"
            try:
                csv_resp = requests.get(csv_url, headers=headers, timeout=15)
                if csv_resp.status_code == 200:
                    df = pd.read_csv(StringIO(csv_resp.text))
                    records = self._parse_dataframe(df)
                    return {"success": True, "data": records}
            except Exception:
                pass

            # Fallback: try pandas.read_html with multiple strategies
            try:
                dfs = pd.read_html(ELORATINGS_URL)
                if dfs:
                    for df in dfs:
                        records = self._parse_dataframe(df)
                        if records:
                            return {"success": True, "data": records}
            except Exception as e:
                self.logger.debug(f"pandas.read_html failed: {e}")

            # If all else fails, provide helpful message
            self.logger.warning(
                "Could not scrape eloratings.net (site uses JavaScript rendering). "
                "Manual options:\n"
                "  1. Export from https://www.eloratings.net/export (if available)\n"
                "  2. Use Selenium/Playwright for JS rendering\n"
                "  3. Download CSV from eloratings.net dashboard manually"
            )
            return {"success": False, "error": "eloratings.net requires JavaScript rendering", "data": []}

        except Exception as e:
            return {"success": False, "error": str(e), "data": []}

    def _parse_dataframe(self, df: pd.DataFrame) -> list:
        """Parse dataframe from CSV or pandas.read_html"""
        records = []
        try:
            # Try to infer columns from the dataframe
            for _, row in df.iterrows():
                team = None
                elo_rating = None
                rank = None

                # Try different column name patterns
                for col in df.columns:
                    col_lower = str(col).lower()
                    if "rank" in col_lower and rank is None:
                        try:
                            rank = int(row[col])
                        except (ValueError, TypeError):
                            pass
                    elif ("team" in col_lower or "name" in col_lower) and team is None:
                        team = str(row[col]).strip()
                    elif "elo" in col_lower and "rating" in col_lower and elo_rating is None:
                        try:
                            elo_rating = float(str(row[col]).replace(",", "."))
                        except (ValueError, TypeError):
                            pass

                if team and elo_rating:
                    record = {
                        "rank": rank or 0,
                        "team": team,
                        "elo_rating": elo_rating,
                        "elo_change": 0.0,
                        "scraped_at": datetime.now().isoformat(),
                    }
                    records.append(record)
        except Exception as e:
            return []

        return records

    def save(self, data: list) -> int:
        """Overwrite elo_current.csv completely (snapshot, not incremental)"""
        if not data:
            return 0

        df = pd.DataFrame(data)
        df.to_csv(ELO_CSV, index=False)
        return len(data)