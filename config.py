import os
from pathlib import Path

# Paths
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")
STATE_FILE = Path("state.json")

# Football-Data.org
FD_API_KEY = os.getenv("FD_API_KEY", "tu_api_key_aqui")
FD_BASE_URL = "https://api.football-data.org/v4"
FD_RATE_LIMIT_PER_MINUTE = 10  # Free tier

# Eloratings.net
ELORATINGS_URL = "https://www.eloratings.net"
ELORATINGS_CSV_URL = "https://www.eloratings.net/export/matches.csv"

# OpenFootball
OPENFOOTBALL_REPO = "https://raw.githubusercontent.com/openfootball/worldcup.json/master"

# Competiciones a ingestar
COMPETITIONS = {
    "WC": {"name": "FIFA World Cup", "seasons": [2014, 2018, 2022, 2026]},
    "CL": {"name": "UEFA Champions League", "seasons": [2023, 2024]},
    "PL": {"name": "Premier League", "seasons": [2023, 2024]},
}

# Control de schedule
FETCH_SCHEDULE = {
    "football_data": "0 1 * * *",      # 1am diario
    "eloratings": "0 2 * * *",         # 2am diario
    "openfootball": "0 3 * * 0",       # 3am cada domingo
}

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)