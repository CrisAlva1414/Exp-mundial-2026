import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


class XGBoostPredictor:

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            eval_metric="mlogloss",
        )
        self.calibrator = CalibratedClassifierCV(self.model, method="isotonic", cv=5)
        self.scaler = StandardScaler()
        self.feature_names = []

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None) -> dict:
        if feature_names:
            self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.calibrator.fit(X_scaled, y)
        return self.evaluate(X_scaled, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.calibrator.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred  = self.calibrator.predict(X)
        y_proba = self.calibrator.predict_proba(X)
        metrics = {
            "accuracy":  float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
            "recall":    float(recall_score(y, y_pred, average="macro", zero_division=0)),
        }
        try:
            metrics["auc_ovo"] = float(roc_auc_score(y, y_proba, multi_class="ovo"))
        except ValueError:
            metrics["auc_ovo"] = None
        return metrics

    def save(self, model_path: str, scaler_path: str):
        with open(model_path, "wb") as f:
            pickle.dump(self.calibrator, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, model_path: str, scaler_path: str):
        instance = cls()
        with open(model_path, "rb") as f:
            instance.calibrator = pickle.load(f)
        with open(scaler_path, "rb") as f:
            instance.scaler = pickle.load(f)
        return instance


class FeatureBuilder:

    COMPETITIONS = ["WC", "CA", "EC", "AFCON", "AFC", "CONCACAF", "NL", "CL", "PL", "PD", "SA", "FR", "OTHER"]

    def __init__(self):
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> list:
        base = [
            "home_form_goals_5", "away_form_goals_5",
            "home_form_pts_5",   "away_form_pts_5",
            "home_goals_avg_all", "away_goals_avg_all",
            "home_conceded_avg",  "away_conceded_avg",
            "home_days_rest",     "away_days_rest",
            "elo_diff",
            "home_advantage",
            "is_neutral",
            "home_win_rate",      "away_win_rate",
            "head2head_home_wins", "head2head_draws",
        ]
        return base + [f"comp_{c}" for c in self.COMPETITIONS]

    def fit_transform(self, df: pd.DataFrame, elo_df: pd.DataFrame = None) -> tuple:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        elo_map = self._build_elo_map(elo_df) if elo_df is not None and not elo_df.empty else {}

        X_list, y_list = [], []

        for idx in range(len(df)):
            row     = df.iloc[idx]
            history = df.iloc[:idx]

            if len(history) < 20:
                continue

            feats = self._build_features(row, history, elo_map)
            X_list.append(feats)

            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
            y_list.append(0 if hg > ag else (1 if hg == ag else 2))

        return np.array(X_list), np.array(y_list), self.feature_names

    def _build_elo_map(self, elo_df: pd.DataFrame) -> dict:
        return dict(zip(elo_df["team"].str.lower(), elo_df["elo_rating"].astype(float)))

    def _build_features(self, match: pd.Series, history: pd.DataFrame, elo_map: dict) -> list:
        home = match["home_team"]
        away = match["away_team"]
        comp = match.get("competition", "OTHER")
        date = match["date"]
        is_neutral = str(match.get("venue", "")).lower() == "neutral"

        h2h = self._head2head(home, away, history, n=10)

        feats = [
            self._goals_avg(home, history, side="home",  n=5),
            self._goals_avg(away, history, side="away",  n=5),
            self._form_pts(home, history, side="home",   n=5),
            self._form_pts(away, history, side="away",   n=5),
            self._goals_avg(home, history, side="home"),
            self._goals_avg(away, history, side="away"),
            self._conceded_avg(home, history, side="home"),
            self._conceded_avg(away, history, side="away"),
            self._days_rest(home, history, ref=date),
            self._days_rest(away, history, ref=date),
            self._elo_diff(home, away, elo_map),
            0.0 if is_neutral else 1.0,
            1.0 if is_neutral else 0.0,
            self._win_rate(home, history, side="home"),
            self._win_rate(away, history, side="away"),
            h2h["home_wins"],
            h2h["draws"],
        ] + self._onehot_competition(comp)

        return feats

    def _goals_avg(self, team: str, history: pd.DataFrame, side: str, n: int = None) -> float:
        col_team  = "home_team" if side == "home" else "away_team"
        col_goals = "home_goals" if side == "home" else "away_goals"
        rows = history[history[col_team] == team]
        if n:
            rows = rows.tail(n)
        return float(rows[col_goals].mean()) if not rows.empty else 1.2

    def _form_pts(self, team: str, history: pd.DataFrame, side: str, n: int = 5) -> float:
        col_team = "home_team" if side == "home" else "away_team"
        rows = history[history[col_team] == team].tail(n)
        pts = 0
        for _, m in rows.iterrows():
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            if side == "home":
                pts += 3 if hg > ag else (1 if hg == ag else 0)
            else:
                pts += 3 if ag > hg else (1 if ag == hg else 0)
        return float(pts)

    def _conceded_avg(self, team: str, history: pd.DataFrame, side: str, n: int = None) -> float:
        if side == "home":
            rows = history[history["home_team"] == team]
            col  = "away_goals"
        else:
            rows = history[history["away_team"] == team]
            col  = "home_goals"
        if n:
            rows = rows.tail(n)
        return float(rows[col].mean()) if not rows.empty else 1.2

    def _win_rate(self, team: str, history: pd.DataFrame, side: str, n: int = 20) -> float:
        col_team = "home_team" if side == "home" else "away_team"
        rows = history[history[col_team] == team].tail(n)
        if rows.empty:
            return 0.33
        wins = sum(
            1 for _, m in rows.iterrows()
            if (side == "home" and m["home_goals"] > m["away_goals"])
            or (side == "away" and m["away_goals"] > m["home_goals"])
        )
        return float(wins / len(rows))

    def _days_rest(self, team: str, history: pd.DataFrame, ref) -> float:
        rows = history[(history["home_team"] == team) | (history["away_team"] == team)].sort_values("date")
        if rows.empty:
            return 30.0
        return float((ref - rows.iloc[-1]["date"]).days)

    def _elo_diff(self, home: str, away: str, elo_map: dict) -> float:
        h = elo_map.get(home.lower(), 1500.0)
        a = elo_map.get(away.lower(), 1500.0)
        return float(h - a)

    def _head2head(self, home: str, away: str, history: pd.DataFrame, n: int = 10) -> dict:
        mask = (
            ((history["home_team"] == home) & (history["away_team"] == away)) |
            ((history["home_team"] == away) & (history["away_team"] == home))
        )
        h2h = history[mask].tail(n)
        hw = draws = 0
        for _, m in h2h.iterrows():
            if m["home_team"] == home:
                if m["home_goals"] > m["away_goals"]:   hw += 1
                elif m["home_goals"] == m["away_goals"]: draws += 1
            else:
                if m["away_goals"] > m["home_goals"]:   hw += 1
                elif m["away_goals"] == m["home_goals"]: draws += 1
        total = len(h2h) or 1
        return {"home_wins": hw / total, "draws": draws / total}

    def _onehot_competition(self, comp: str) -> list:
        return [1.0 if comp == c else 0.0 for c in self.COMPETITIONS]

    def transform_single(
        self,
        home_team: str,
        away_team: str,
        competition: str,
        date,
        history: pd.DataFrame,
        elo_map: dict = None,
    ) -> np.ndarray:
        row = pd.Series({
            "home_team":  home_team,
            "away_team":  away_team,
            "competition": competition,
            "date":       pd.to_datetime(date),
            "venue":      "neutral",
            "home_goals": 0,
            "away_goals": 0,
        })
        feats = self._build_features(row, history, elo_map or {})
        return np.array(feats)