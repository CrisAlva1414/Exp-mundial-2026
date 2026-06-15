import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class XGBoostPredictor:
    """XGBoost classifier with calibration for multi-class prediction"""

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=1,
        )
        self.calibrator = CalibratedClassifierCV(
            self.model,
            method="isotonic",
            cv=5
        )
        self.scaler = StandardScaler()

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train XGBoost with calibration
        X: (n_samples, n_features) - feature matrix
        y: (n_samples,) - labels [0=home_win, 1=draw, 2=away_win]
        """
        X_scaled = self.scaler.fit_transform(X)
        self.calibrator.fit(X_scaled, y)
        return self.evaluate(X_scaled, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for 3 classes
        Return: (n_samples, 3) array with [P(home_win), P(draw), P(away_win)]
        """
        X_scaled = self.scaler.transform(X)
        return self.calibrator.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model on given data"""
        y_pred = self.calibrator.predict(X)
        y_proba = self.calibrator.predict_proba(X)

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
            "auc_ovo": float(roc_auc_score(y, y_proba, multi_class="ovo", zero_division=0)),
        }

    def save(self, model_path: str, scaler_path: str):
        """Save model and scaler"""
        with open(model_path, "wb") as f:
            pickle.dump(self.calibrator, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, model_path: str, scaler_path: str):
        """Load model and scaler"""
        instance = cls()
        with open(model_path, "rb") as f:
            instance.calibrator = pickle.load(f)
        with open(scaler_path, "rb") as f:
            instance.scaler = pickle.load(f)
        return instance


class FeatureBuilder:
    """Build features for XGBoost from historical match data"""

    def __init__(self):
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Build features from historical matches
        df: matches.csv with columns: date, home_team, away_team, home_goals, away_goals, competition, venue, etc.
        Returns: (X, y, feature_names)
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        X_list = []
        y_list = []

        for idx, row in df.iterrows():
            features = self._build_features(row, df.iloc[:idx])
            if features is not None:
                X_list.append(features["features"])
                y_list.append(features["target"])

        if not X_list:
            return np.array([]), np.array([]), []

        X = np.array(X_list)
        y = np.array(y_list)

        self.feature_names = features["feature_names"]

        return X, y, self.feature_names

    def _build_features(self, match: pd.Series, history: pd.DataFrame) -> dict | None:
        """Build features for a single match"""
        home = match["home_team"]
        away = match["away_team"]
        competition = match.get("competition", "")
        date = match["date"]
        is_neutral = match.get("venue", "") == "neutral"

        if history.empty:
            return None

        # Form features (last 5 matches)
        home_form_5 = self._calculate_form(home, history, last_n=5, as_home=True)
        away_form_5 = self._calculate_form(away, history, last_n=5, as_home=False)

        # Form recent (W=3, D=1, L=0)
        home_form_recent = self._calculate_form_recent(home, history, last_n=3, as_home=True)
        away_form_recent = self._calculate_form_recent(away, history, last_n=3, as_home=False)

        # Goals averages
        home_goals_avg = self._calculate_goals_avg(home, history, as_home=True)
        away_goals_conceded_avg = self._calculate_goals_conceded_avg(away, history, as_home=False)

        # Days since last match
        home_days_since = self._days_since_last_match(home, history, as_home=True, current_date=date)
        away_days_since = self._days_since_last_match(away, history, as_home=False, current_date=date)

        # Home advantage
        home_advantage = 0 if is_neutral else 1

        # ELO diff (hardcoded 0 for now, would need external source)
        elo_diff = 0

        # Binary features
        is_knockout = 0  # Would need match-level info
        is_neutral_venue = 1 if is_neutral else 0

        # One-hot encoding for competition
        comp_features = self._onehot_competition(competition)

        # Target (home_goals vs away_goals)
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])
        if home_goals > away_goals:
            target = 0  # Home win
        elif home_goals == away_goals:
            target = 1  # Draw
        else:
            target = 2  # Away win

        features = [
            home_form_5,
            away_form_5,
            home_form_recent,
            away_form_recent,
            home_goals_avg,
            away_goals_conceded_avg,
            home_days_since,
            away_days_since,
            home_advantage,
            elo_diff,
            is_knockout,
            is_neutral_venue,
        ] + comp_features

        feature_names = [
            "home_form_5",
            "away_form_5",
            "home_form_recent",
            "away_form_recent",
            "home_goals_avg",
            "away_goals_conceded_avg",
            "home_days_since",
            "away_days_since",
            "home_advantage",
            "elo_diff",
            "is_knockout",
            "is_neutral_venue",
        ] + [f"comp_{c}" for c in ["WC", "CL", "PL", "PD", "SA", "OTHER"]]

        return {
            "features": features,
            "target": target,
            "feature_names": feature_names,
        }

    def _calculate_form(self, team: str, history: pd.DataFrame, last_n: int = 5, as_home: bool = True) -> float:
        """Calculate average goals in last N matches"""
        if as_home:
            matches = history[history["home_team"] == team].tail(last_n)
            if matches.empty:
                return 0.0
            return float(matches["home_goals"].mean())
        else:
            matches = history[history["away_team"] == team].tail(last_n)
            if matches.empty:
                return 0.0
            return float(matches["away_goals"].mean())

    def _calculate_form_recent(self, team: str, history: pd.DataFrame, last_n: int = 3, as_home: bool = True) -> float:
        """Calculate form points: W=3, D=1, L=0"""
        points = 0
        if as_home:
            matches = history[history["home_team"] == team].tail(last_n)
            for _, m in matches.iterrows():
                if m["home_goals"] > m["away_goals"]:
                    points += 3
                elif m["home_goals"] == m["away_goals"]:
                    points += 1
        else:
            matches = history[history["away_team"] == team].tail(last_n)
            for _, m in matches.iterrows():
                if m["away_goals"] > m["home_goals"]:
                    points += 3
                elif m["away_goals"] == m["home_goals"]:
                    points += 1
        return float(points)

    def _calculate_goals_avg(self, team: str, history: pd.DataFrame, as_home: bool = True) -> float:
        """Average goals per match in the season"""
        if as_home:
            matches = history[history["home_team"] == team]
            if matches.empty:
                return 0.0
            return float(matches["home_goals"].mean())
        else:
            matches = history[history["home_team"] == team]
            if matches.empty:
                return 0.0
            return float(matches["away_goals"].mean())

    def _calculate_goals_conceded_avg(self, team: str, history: pd.DataFrame, as_home: bool = True) -> float:
        """Average goals conceded per match"""
        if as_home:
            matches = history[history["away_team"] == team]
            if matches.empty:
                return 0.0
            return float(matches["away_goals"].mean())
        else:
            matches = history[history["home_team"] == team]
            if matches.empty:
                return 0.0
            return float(matches["home_goals"].mean())

    def _days_since_last_match(self, team: str, history: pd.DataFrame, as_home: bool = True, current_date=None) -> float:
        """Days since last match"""
        if as_home:
            matches = history[history["home_team"] == team].sort_values("date")
        else:
            matches = history[history["away_team"] == team].sort_values("date")

        if matches.empty:
            return 0.0

        last_date = matches.iloc[-1]["date"]
        if current_date is None:
            current_date = history["date"].max()

        delta = (current_date - last_date).days
        return float(delta)

    def _onehot_competition(self, comp: str) -> list:
        """One-hot encoding for competition"""
        comps = ["WC", "CL", "PL", "PD", "SA", "OTHER"]
        return [1.0 if comp == c else 0.0 for c in comps]

    def transform(self, home_team: str, away_team: str, competition: str, date, df_history: pd.DataFrame) -> np.ndarray:
        """Transform a single match to features"""
        match = pd.Series({
            "home_team": home_team,
            "away_team": away_team,
            "competition": competition,
            "date": date,
            "venue": "unknown",
            "home_goals": 0,
            "away_goals": 0,
        })

        features = self._build_features(match, df_history)
        if features is None:
            return np.zeros(len(self.feature_names))

        return np.array(features["features"])
