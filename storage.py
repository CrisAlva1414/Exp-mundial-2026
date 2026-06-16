import json
import pandas as pd
from config import MATCHES_CSV, ELO_CSV


def load_matches() -> pd.DataFrame:
    if MATCHES_CSV.exists():
        return pd.read_csv(MATCHES_CSV, parse_dates=["date"])
    return pd.DataFrame()


def load_elo() -> pd.DataFrame:
    if ELO_CSV.exists():
        return pd.read_csv(ELO_CSV)
    return pd.DataFrame()


def elo_map(elo_df: pd.DataFrame = None) -> dict:
    df = elo_df if elo_df is not None else load_elo()
    if df.empty:
        return {}
    return dict(zip(df["team"].str.lower(), df["elo_rating"].astype(float)))