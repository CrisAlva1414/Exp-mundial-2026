import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from model import XGBoostPredictor, FeatureBuilder


class PoissonSimulator:
    """Simulate football match scores using Poisson distribution"""

    def estimate_lambda(self, p_win: float, p_draw: float) -> tuple[float, float]:
        """
        Estimate Poisson parameters from XGBoost probabilities
        Uses Dixon-Coles (1997) methodology
        """
        lambda_home = 1.5 + (0.5 * (p_win - p_draw * 0.5))
        lambda_away = 1.5 - (0.5 * (p_win - p_draw * 0.5))
        return max(lambda_home, 0.1), max(lambda_away, 0.1)

    def simulate(self, lambda_home: float, lambda_away: float, n_sims: int = 10000) -> dict:
        """
        Simulate match outcomes using Poisson distribution
        Returns distribution of goals and probabilities
        """
        home_goals = np.random.poisson(lambda_home, n_sims)
        away_goals = np.random.poisson(lambda_away, n_sims)

        # Count outcomes
        outcomes = np.where(
            home_goals > away_goals, 0,
            np.where(home_goals == away_goals, 1, 2)
        )

        outcome_counts = np.bincount(outcomes, minlength=3)
        prob_hw, prob_d, prob_aw = outcome_counts / n_sims

        # Most likely score
        scores = list(zip(home_goals, away_goals))
        score_counts = Counter(scores)
        most_likely = score_counts.most_common(1)[0][0]

        # Top 10 scores by probability
        score_dist = {k: v / n_sims for k, v in score_counts.most_common(10)}

        return {
            "prob_home": float(prob_hw),
            "prob_draw": float(prob_d),
            "prob_away": float(prob_aw),
            "most_likely_score": most_likely,
            "score_distribution": score_dist,
        }


class FootballPredictor:
    """End-to-end predictor: XGBoost + Poisson"""

    def __init__(self, model_path: str, scaler_path: str):
        self.xgb = XGBoostPredictor.load(model_path, scaler_path)
        self.poisson = PoissonSimulator()
        self.feature_builder = FeatureBuilder()

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        competition: str,
        date,
        df_history: pd.DataFrame
    ) -> dict:
        """
        Full prediction pipeline for a single match
        Returns: probabilities, score distribution, and explanation
        """
        # Build features
        X = self.feature_builder.transform(home_team, away_team, competition, date, df_history)
        X = X.reshape(1, -1)

        # XGBoost prediction
        probs = self.xgb.predict_proba(X)[0]
        p_home = probs[0]
        p_draw = probs[1]
        p_away = probs[2]

        # Poisson simulation
        lambda_home, lambda_away = self.poisson.estimate_lambda(p_home, p_draw)
        poisson_result = self.poisson.simulate(lambda_home, lambda_away)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "competition": competition,
            "date": str(date),
            "xgb_probs": {
                "home_win": float(p_home),
                "draw": float(p_draw),
                "away_win": float(p_away),
            },
            "poisson_simulation": poisson_result,
            "timestamp": datetime.now().isoformat(),
        }
