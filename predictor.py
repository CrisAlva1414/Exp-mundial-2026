import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

from config import MATCHES_CSV, ELO_CSV, MODEL_DIR
from model import XGBoostPredictor, FeatureBuilder

MODEL_PATH  = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


class PoissonSimulator:

    def estimate_lambdas(self, p_home: float, p_draw: float) -> tuple[float, float]:
        # Dixon-Coles (1997): ajuste de expected goals desde probabilidades
        lambda_home = 1.5 + 0.5 * (p_home - p_draw * 0.5)
        lambda_away = 1.5 - 0.5 * (p_home - p_draw * 0.5)
        return max(lambda_home, 0.1), max(lambda_away, 0.1)

    def simulate(self, lambda_home: float, lambda_away: float, n: int = 10_000) -> dict:
        hg = np.random.poisson(lambda_home, n)
        ag = np.random.poisson(lambda_away, n)

        outcomes = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
        counts   = np.bincount(outcomes, minlength=3)
        p_hw, p_d, p_aw = counts / n

        score_dist   = Counter(zip(hg.tolist(), ag.tolist()))
        top_scores   = {f"{h}-{a}": v / n for (h, a), v in score_dist.most_common(10)}
        most_likely  = score_dist.most_common(1)[0][0]

        return {
            "prob_home": float(p_hw),
            "prob_draw": float(p_d),
            "prob_away": float(p_aw),
            "most_likely_score": f"{most_likely[0]}-{most_likely[1]}",
            "top_scores": top_scores,
        }


class FootballPredictor:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

        self.xgb     = XGBoostPredictor.load(str(MODEL_PATH), str(SCALER_PATH))
        self.poisson = PoissonSimulator()
        self.builder = FeatureBuilder()
        self._matches = self._load_matches()
        self._elo_map = self._load_elo()

    def _load_matches(self) -> pd.DataFrame:
        if not MATCHES_CSV.exists():
            raise FileNotFoundError(f"{MATCHES_CSV} not found. Run pipeline.py first.")
        df = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _load_elo(self) -> dict:
        if not ELO_CSV.exists():
            return {}
        elo = pd.read_csv(ELO_CSV)
        return dict(zip(elo["team"].str.lower(), elo["elo_rating"].astype(float)))

    def predict(
        self,
        home_team: str,
        away_team: str,
        competition: str = "WC",
        date=None,
    ) -> dict:
        if date is None:
            date = datetime.today()

        date = pd.to_datetime(date)

        # Historia disponible hasta la fecha del partido
        history = self._matches[self._matches["date"] < date]

        X = self.builder.transform_single(
            home_team, away_team, competition, date, history, self._elo_map
        ).reshape(1, -1)

        probs = self.xgb.predict_proba(X)[0]
        p_home, p_draw, p_away = probs

        lh, la = self.poisson.estimate_lambdas(float(p_home), float(p_draw))
        sim = self.poisson.simulate(lh, la)

        elo_h = self._elo_map.get(home_team.lower(), None)
        elo_a = self._elo_map.get(away_team.lower(), None)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "competition": competition,
            "date": date.strftime("%Y-%m-%d"),
            "xgb": {
                "home_win": round(float(p_home), 4),
                "draw":     round(float(p_draw), 4),
                "away_win": round(float(p_away), 4),
            },
            "simulation": sim,
            "elo": {
                "home": int(elo_h) if elo_h else None,
                "away": int(elo_a) if elo_a else None,
                "diff": int(elo_h - elo_a) if (elo_h and elo_a) else None,
            },
            "predicted_at": datetime.now().isoformat(),
        }

    def predict_group(self, matches: list[dict]) -> list[dict]:
        return [self.predict(**m) for m in matches]


if __name__ == "__main__":
    import json

    predictor = FootballPredictor()
    result = predictor.predict("Argentina", "France", competition="WC")
    print(json.dumps(result, indent=2))