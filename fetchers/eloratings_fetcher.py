import requests
import pandas as pd

from datetime import datetime
from pathlib import Path

from config import ELO_CSV
from .base_fetcher import BaseFetcher


class EloratingsFetcher(BaseFetcher):

    WORLD_URL = "https://www.eloratings.net/World.tsv"
    TEAMS_URL = "https://www.eloratings.net/en.teams.tsv"

    # columnas World.tsv
    RANK = 0
    LOCAL_RANK = 1
    COUNTRY_CODE = 2
    RATING = 3

    BEST_RANK = 4
    BEST_RATING = 5
    WORST_RANK = 6
    WORST_RATING = 7

    AVG_RANK = 8
    AVG_RATING = 9

    RANK_CHANGE = 10
    RATING_CHANGE = 11

    RANK_CHANGE_3M = 14
    RATING_CHANGE_3M = 15

    RANK_CHANGE_6M = 16
    RATING_CHANGE_6M = 17

    RANK_CHANGE_1Y = 18
    RATING_CHANGE_1Y = 19

    RANK_CHANGE_2Y = 20
    RATING_CHANGE_2Y = 21

    MATCHES = 22
    WINS = 23
    DRAWS = 24
    LOSSES = 25

    HOME_MATCHES = 26
    AWAY_MATCHES = 27
    NEUTRAL_MATCHES = 28

    GOALS_FOR = 29
    GOALS_AGAINST = 30

    def __init__(self, data_dir: Path, logger):
        super().__init__("eloratings", data_dir, logger)

    def fetch(self) -> dict:

        try:

            self.rate_limit()

            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; football-predictor/1.0)"
            }

            # Descargar World.tsv
            world_resp = requests.get(
                self.WORLD_URL,
                headers=headers,
                timeout=15
            )
            world_resp.raise_for_status()
            world_resp.encoding = "utf-8"

            # Descargar nombres de equipos
            teams_resp = requests.get(
                self.TEAMS_URL,
                headers=headers,
                timeout=15
            )
            teams_resp.raise_for_status()
            teams_resp.encoding = "utf-8"

            team_names = {}

            for line in teams_resp.text.splitlines():

                parts = line.split("\t")

                if len(parts) >= 2:
                    code = parts[0].strip()
                    name = parts[1].strip()

                    team_names[code] = name

            # Parsear World.tsv
            records = []

            for line in world_resp.text.splitlines():

                cols = line.split("\t")

                if len(cols) < 31:
                    continue

                code = cols[self.COUNTRY_CODE]

                record = {

                    "rank": int(cols[self.RANK]),
                    "local_rank": int(cols[self.LOCAL_RANK]),

                    "team": team_names.get(code, code),
                    "country_code": code,

                    "elo_rating": int(cols[self.RATING]),

                    "best_rank": int(cols[self.BEST_RANK]),
                    "best_rating": int(cols[self.BEST_RATING]),

                    "worst_rank": int(cols[self.WORST_RANK]),
                    "worst_rating": int(cols[self.WORST_RATING]),

                    "average_rank": int(cols[self.AVG_RANK]),
                    "average_rating": int(cols[self.AVG_RATING]),

                    "rank_change": cols[self.RANK_CHANGE],
                    "elo_change": cols[self.RATING_CHANGE],

                    "rank_change_3m": cols[self.RANK_CHANGE_3M],
                    "elo_change_3m": cols[self.RATING_CHANGE_3M],

                    "rank_change_6m": cols[self.RANK_CHANGE_6M],
                    "elo_change_6m": cols[self.RATING_CHANGE_6M],

                    "rank_change_1y": cols[self.RANK_CHANGE_1Y],
                    "elo_change_1y": cols[self.RATING_CHANGE_1Y],

                    "rank_change_2y": cols[self.RANK_CHANGE_2Y],
                    "elo_change_2y": cols[self.RATING_CHANGE_2Y],

                    "matches": int(cols[self.MATCHES]),
                    "wins": int(cols[self.WINS]),
                    "draws": int(cols[self.DRAWS]),
                    "losses": int(cols[self.LOSSES]),

                    "home_matches": int(cols[self.HOME_MATCHES]),
                    "away_matches": int(cols[self.AWAY_MATCHES]),
                    "neutral_matches": int(cols[self.NEUTRAL_MATCHES]),

                    "goals_for": int(cols[self.GOALS_FOR]),
                    "goals_against": int(cols[self.GOALS_AGAINST]),

                    "scraped_at": datetime.now().isoformat()
                }

                records.append(record)

            return {
                "success": True,
                "data": records
            }

        except Exception as e:

            self.logger.exception("Error fetching ELO ratings")

            return {
                "success": False,
                "error": str(e),
                "data": []
            }

    def save(self, data: list) -> int:
        if not data:
            return 0

        df = pd.DataFrame(data)

        df.to_csv(
            ELO_CSV,
            index=False
        )

        return len(df)