import os
from pathlib import Path

# Paths
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
LOGS_DIR = Path("logs")
STATE_FILE = Path("state.json")
ELO_CSV = DATA_DIR / "elo_current.csv"
MATCHES_CSV = DATA_DIR / "matches.csv"

# Football-Data.org
FD_API_KEY = os.getenv("FD_API_KEY", "tu_api_key_aqui")
FD_BASE_URL = "https://api.football-data.org/v4"
FD_RATE_LIMIT_PER_MINUTE = 10  # Free tier

# Kaggle dataset (martj42/international-football-results-from-1872-to-2017)
KAGGLE_RESULTS_CSV = RAW_DATA_DIR / "results.csv"

# Competiciones para football-data.org
COMPETITIONS = {
    "WC": {"name": "FIFA World Cup", "seasons": [2014, 2018, 2022, 2026]},
    "CL": {"name": "UEFA Champions League", "seasons": [2023, 2024]},
    "PL": {"name": "Premier League", "seasons": [2023, 2024]},
    "PD": {"name": "La Liga", "seasons": [2023, 2024]},
    "SA": {"name": "Serie A", "seasons": [2023, 2024]},
}

# Crear directorios
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)