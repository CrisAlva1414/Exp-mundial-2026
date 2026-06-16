import logging
import pandas as pd
from pathlib import Path
from config import DATA_DIR, MATCHES_CSV, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SCHEMA = ["match_id", "source", "competition", "season", "date",
          "home_team", "away_team", "home_goals", "away_goals", "venue"]

INTL_KEYWORDS = ["World Cup", "Copa Am", "Copa America", "UEFA Euro", "African Cup",
                 "African Nations", "AFC Asian", "Gold Cup", "Nations League",
                 "Confederations Cup", "CONMEBOL", "Friendly"]

# Incluimos friendlies para tener más historia por equipo, pero los marcamos
EXCLUDE_TOURNAMENTS = set()


def _tournament_to_code(t: str) -> str:
    t_l = t.lower()
    if "world cup" in t_l:       return "WC"
    if "copa am" in t_l:         return "CA"
    if "euro" in t_l:            return "EC"
    if "african" in t_l or "afcon" in t_l or "nations cup" in t_l: return "AFCON"
    if "asian" in t_l:           return "AFC"
    if "gold cup" in t_l:        return "CONCACAF"
    if "nations league" in t_l:  return "NL"
    if "friendly" in t_l:        return "FR"
    if "confederations" in t_l:  return "CC"
    return "OTHER"


def _load_kaggle_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"Not found: {path.name}")
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["home_score"] >= 0]
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})

    df["competition"] = df["tournament"].apply(_tournament_to_code)
    df["season"]      = df["date"].dt.year.astype(str)
    df["source"]      = "kaggle"
    df["venue"]       = df.apply(
        lambda r: "neutral" if r.get("neutral") else r.get("city", "unknown"), axis=1
    )
    df["match_id"] = (
        "kg_"
        + df["date"].dt.strftime("%Y%m%d")
        + "_"
        + df["home_team"].str.replace(" ", "_", regex=False)
        + "_"
        + df["away_team"].str.replace(" ", "_", regex=False)
    )
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[SCHEMA].copy()


def _load_openfootball(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"Not found: {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    return df[SCHEMA].copy()


def _load_football_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning(f"Not found: {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    return df[SCHEMA].copy()


def _resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    PRIORITY = {"football_data": 0, "openfootball": 1, "kaggle": 2}
    df["_p"] = df["source"].map(PRIORITY).fillna(99)
    df["_key"] = (
        df["date"]
        + "|"
        + df["home_team"].str.lower().str.strip()
        + "|"
        + df["away_team"].str.lower().str.strip()
    )
    df = df.sort_values("_p").drop_duplicates(subset=["_key"], keep="first")
    return df.drop(columns=["_p", "_key"])


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df[df["home_goals"] >= 0]
    df = df[df["away_goals"] >= 0]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df[df["date"].notna()]
    df = df[df["home_team"].str.strip() != ""]
    df = df[df["away_team"].str.strip() != ""]
    logger.info(f"Validation: {before} → {len(df)} rows ({before - len(df)} dropped)")
    return df


def run_pipeline() -> pd.DataFrame:
    logger.info("Starting merge pipeline...")

    sources = []

    kg = _load_kaggle_results(DATA_DIR / "kaggle_results_current.csv")
    if not kg.empty:
        logger.info(f"Kaggle results:  {len(kg):,} rows")
        sources.append(kg)

    of = _load_openfootball(DATA_DIR / "openfootball_current.csv")
    if not of.empty:
        logger.info(f"OpenFootball:    {len(of):,} rows")
        sources.append(of)

    fd = _load_football_data(DATA_DIR / "datafootball_current.csv")
    if not fd.empty:
        logger.info(f"Football-Data:   {len(fd):,} rows")
        sources.append(fd)

    if not sources:
        logger.error("No data sources found.")
        return pd.DataFrame()

    df = pd.concat(sources, ignore_index=True)
    logger.info(f"Combined (raw):  {len(df):,} rows")

    df = _validate(df)
    df = _resolve_duplicates(df)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(MATCHES_CSV, index=False)
    logger.info(f"Saved {len(df):,} matches → {MATCHES_CSV}")

    for src, grp in df.groupby("source"):
        logger.info(f"  {src}: {len(grp):,} matches")

    return df


if __name__ == "__main__":
    df = run_pipeline()
    if not df.empty:
        print(f"\nmatches.csv → {len(df):,} rows")
        print(df.groupby(["competition"])["match_id"].count().sort_values(ascending=False).to_string())