import json
import pandas as pd
from pathlib import Path
from config import MATCHES_CSV, ELO_CSV, STATE_FILE


def load_matches() -> pd.DataFrame:
    if MATCHES_CSV.exists():
        return pd.read_csv(MATCHES_CSV)
    return pd.DataFrame()


def save_matches(df: pd.DataFrame):
    if df.empty:
        return

    df = df.drop_duplicates(subset=["match_id"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(MATCHES_CSV, index=False)


def load_elo() -> pd.DataFrame:
    if ELO_CSV.exists():
        return pd.read_csv(ELO_CSV)
    return pd.DataFrame()


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
