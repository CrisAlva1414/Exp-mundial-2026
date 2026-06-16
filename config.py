import os
from pathlib import Path

DATA_DIR    = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
LOGS_DIR    = Path("logs")
MODEL_DIR   = Path("model")

ELO_CSV     = DATA_DIR / "elo_current.csv"
MATCHES_CSV = DATA_DIR / "matches.csv"
WC2026_JSON = DATA_DIR / "worldcup.json"

FD_API_KEY  = os.getenv("FD_API_KEY", "")
FD_BASE_URL = "https://api.football-data.org/v4"

DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)