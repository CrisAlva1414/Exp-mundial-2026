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
        return self.calibrator.predict_proba(self.scaler.transform(X))

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
    """
    Vectorized feature engineering — O(n) usando expanding/rolling sobre
    tablas pivoteadas por equipo, sin loops fila por fila.
    """

    COMPETITIONS = ["WC", "CA", "EC", "AFCON", "AFC", "CONCACAF", "NL", "CL", "PL", "PD", "SA", "FR", "OTHER"]
    WINDOW = 5   # partidos recientes para forma
    MIN_ROWS = 5  # mínimo historial para incluir una fila

    def __init__(self):
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> list:
        base = [
            "home_form_goals_5", "away_form_goals_5",
            "home_form_pts_5",   "away_form_pts_5",
            "home_goals_avg",    "away_goals_avg",
            "home_conceded_avg", "away_conceded_avg",
            "home_win_rate",     "away_win_rate",
            "elo_diff",
            "home_advantage",
            "is_neutral",
            "head2head_hw_rate", "head2head_draw_rate",
        ]
        return base + [f"comp_{c}" for c in self.COMPETITIONS]

    # ------------------------------------------------------------------ #
    #  Tabla de stats por equipo computada de forma vectorizada           #
    # ------------------------------------------------------------------ #

    def _build_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Para cada partido, calcula las stats del equipo local y visitante
        usando únicamente información de partidos ANTERIORES (shift(1) evita leakage).
        Devuelve un DataFrame con índice alineado a df.
        """
        # Expandir a formato long: una fila por equipo por partido
        home = df[["date", "home_team", "home_goals", "away_goals", "competition"]].copy()
        home.columns = ["date", "team", "gf", "ga", "competition"]
        home["is_home"] = True

        away = df[["date", "away_team", "away_goals", "home_goals", "competition"]].copy()
        away.columns = ["date", "team", "gf", "ga", "competition"]
        away["is_home"] = False

        long = pd.concat([home, away], ignore_index=True)
        long = long.sort_values(["team", "date"]).reset_index(drop=True)

        long["win"]  = (long["gf"] > long["ga"]).astype(float)
        long["draw"] = (long["gf"] == long["ga"]).astype(float)
        long["pts"]  = long["win"] * 3 + long["draw"]

        # Expanding mean (todos los partidos previos) — shift(1) para no incluir el actual
        grp = long.groupby("team")
        long["avg_gf"]    = grp["gf"].transform(lambda x: x.shift(1).expanding().mean())
        long["avg_ga"]    = grp["ga"].transform(lambda x: x.shift(1).expanding().mean())
        long["win_rate"]  = grp["win"].transform(lambda x: x.shift(1).expanding().mean())

        # Rolling window para forma reciente
        long["form_gf"]  = grp["gf"].transform(lambda x: x.shift(1).rolling(self.WINDOW, min_periods=1).mean())
        long["form_pts"] = grp["pts"].transform(lambda x: x.shift(1).rolling(self.WINDOW, min_periods=1).sum())

        return long

    def _build_h2h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Head-to-head: para cada partido, busca los últimos N enfrentamientos
        entre los mismos dos equipos (en cualquier dirección).
        Devuelve columnas h2h_hw_rate y h2h_draw_rate con índice = df.index.
        """
        N = 8
        results = []

        # Crear clave canónica de par (team_a < team_b alfabéticamente)
        df = df.copy().reset_index(drop=True)
        df["pair"] = df.apply(
            lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
        )

        pair_groups = df.groupby("pair")

        for pair, group in pair_groups:
            group = group.sort_values("date").reset_index()
            hw_rates, draw_rates = [], []

            for i in range(len(group)):
                past = group.iloc[:i]
                if past.empty:
                    hw_rates.append(0.33)
                    draw_rates.append(0.25)
                    continue
                past = past.tail(N)
                home_team = group.iloc[i]["home_team"]
                hw = 0
                dr = 0
                for _, m in past.iterrows():
                    if m["home_team"] == home_team:
                        if m["home_goals"] > m["away_goals"]:   hw += 1
                        elif m["home_goals"] == m["away_goals"]: dr += 1
                    else:
                        if m["away_goals"] > m["home_goals"]:   hw += 1
                        elif m["away_goals"] == m["home_goals"]: dr += 1
                n = len(past)
                hw_rates.append(hw / n)
                draw_rates.append(dr / n)

            group["h2h_hw_rate"]   = hw_rates
            group["h2h_draw_rate"] = draw_rates
            results.append(group[["index", "h2h_hw_rate", "h2h_draw_rate"]])

        h2h_df = pd.concat(results).set_index("index").sort_index()
        return h2h_df

    def fit_transform(self, df: pd.DataFrame, elo_df: pd.DataFrame = None) -> tuple:
        import time
        t0 = time.time()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        elo_map = {}
        if elo_df is not None and not elo_df.empty:
            elo_map = dict(zip(elo_df["team"].str.lower(), elo_df["elo_rating"].astype(float)))

        print(f"  Building team stats (vectorized)...", flush=True)
        stats = self._build_team_stats(df)

        # Separar stats por rol
        home_stats = stats[stats["is_home"]].set_index("date")
        away_stats = stats[~stats["is_home"]].set_index("date")

        # Re-indexar alineado a df
        df_idx = df.reset_index(drop=True)

        def get_stat(role_stats, team_col, stat_col, df_ref):
            # Merge por posición: equipo en fila i del df original
            tmp = df_ref[[team_col, "date"]].copy()
            tmp.columns = ["team", "date"]
            merged = tmp.merge(
                role_stats[["team", stat_col]].reset_index(),
                on=["date", "team"],
                how="left",
            )
            # Puede haber duplicados si un equipo jugó dos partidos el mismo día
            merged = merged.groupby(merged.index)[stat_col].first()
            return merged.values

        print(f"  Merging stats per team...", flush=True)

        # Stats por equipo local
        hs = stats[stats["is_home"]].copy()
        hs = hs.rename(columns={"team": "home_team"})
        hs["_pos"] = hs.groupby(["date", "home_team"]).cumcount()

        as_ = stats[~stats["is_home"]].copy()
        as_ = as_.rename(columns={"team": "away_team"})
        as_["_pos"] = as_.groupby(["date", "away_team"]).cumcount()

        df_idx["_pos"] = df_idx.groupby(["date", "home_team"]).cumcount()

        df_m = df_idx.merge(
            hs[["date", "home_team", "avg_gf", "avg_ga", "win_rate", "form_gf", "form_pts", "_pos"]].rename(columns={
                "avg_gf": "h_avg_gf", "avg_ga": "h_avg_ga", "win_rate": "h_win_rate",
                "form_gf": "h_form_gf", "form_pts": "h_form_pts"
            }),
            on=["date", "home_team", "_pos"], how="left"
        ).merge(
            as_[["date", "away_team", "avg_gf", "avg_ga", "win_rate", "form_gf", "form_pts", "_pos"]].rename(columns={
                "avg_gf": "a_avg_gf", "avg_ga": "a_avg_ga", "win_rate": "a_win_rate",
                "form_gf": "a_form_gf", "form_pts": "a_form_pts"
            }),
            on=["date", "away_team", "_pos"], how="left"
        )

        print(f"  Building H2H stats...", flush=True)
        h2h = self._build_h2h(df_idx)
        df_m = df_m.join(h2h[["h2h_hw_rate", "h2h_draw_rate"]])

        print(f"  Building ELO and competition features...", flush=True)
        df_m["elo_home"] = df_m["home_team"].str.lower().map(elo_map).fillna(1500.0)
        df_m["elo_away"] = df_m["away_team"].str.lower().map(elo_map).fillna(1500.0)
        df_m["elo_diff"] = df_m["elo_home"] - df_m["elo_away"]

        df_m["is_neutral"]     = (df_m["venue"].str.lower() == "neutral").astype(float)
        df_m["home_advantage"] = 1.0 - df_m["is_neutral"]

        for c in self.COMPETITIONS:
            df_m[f"comp_{c}"] = (df_m["competition"] == c).astype(float)

        # Target
        df_m["result"] = np.where(
            df_m["home_goals"] > df_m["away_goals"], 0,
            np.where(df_m["home_goals"] == df_m["away_goals"], 1, 2)
        )

        # Eliminar filas con NaN en features clave (primeros partidos de equipos nuevos)
        feature_cols = [
            "h_form_gf", "a_form_gf", "h_form_pts", "a_form_pts",
            "h_avg_gf",  "a_avg_gf",  "h_avg_ga",   "a_avg_ga",
            "h_win_rate","a_win_rate", "elo_diff",
            "home_advantage", "is_neutral",
            "h2h_hw_rate", "h2h_draw_rate",
        ] + [f"comp_{c}" for c in self.COMPETITIONS]

        df_m = df_m.dropna(subset=["h_form_gf", "a_form_gf", "h_avg_gf", "a_avg_gf"])

        X = df_m[feature_cols].values.astype(np.float32)
        y = df_m["result"].values.astype(int)

        print(f"  Feature matrix built in {time.time()-t0:.1f}s — shape {X.shape}", flush=True)
        return X, y, self.feature_names

    def transform_single(
        self,
        home_team: str,
        away_team: str,
        competition: str,
        date,
        history: pd.DataFrame,
        elo_map: dict = None,
    ) -> np.ndarray:
        elo_map = elo_map or {}
        date = pd.to_datetime(date)

        def team_stats(team, side):
            if side == "home":
                rows = history[history["home_team"] == team].sort_values("date")
                gf_col, ga_col = "home_goals", "away_goals"
            else:
                rows = history[history["away_team"] == team].sort_values("date")
                gf_col, ga_col = "away_goals", "home_goals"

            if rows.empty:
                return {"avg_gf": 1.2, "avg_ga": 1.2, "form_gf": 1.2, "form_pts": 4.5, "win_rate": 0.33}

            gf = rows[gf_col]
            ga = rows[ga_col]
            wins  = (gf > ga).astype(float)
            draws = (gf == ga).astype(float)
            pts   = wins * 3 + draws

            return {
                "avg_gf":   float(gf.mean()),
                "avg_ga":   float(ga.mean()),
                "form_gf":  float(gf.tail(self.WINDOW).mean()),
                "form_pts": float(pts.tail(self.WINDOW).sum()),
                "win_rate": float(wins.mean()),
            }

        hs = team_stats(home_team, "home")
        as_ = team_stats(away_team, "away")

        # H2H
        mask = (
            ((history["home_team"] == home_team) & (history["away_team"] == away_team)) |
            ((history["home_team"] == away_team) & (history["away_team"] == home_team))
        )
        h2h = history[mask].tail(8)
        hw = dr = 0
        for _, m in h2h.iterrows():
            if m["home_team"] == home_team:
                if m["home_goals"] > m["away_goals"]:    hw += 1
                elif m["home_goals"] == m["away_goals"]: dr += 1
            else:
                if m["away_goals"] > m["home_goals"]:    hw += 1
                elif m["away_goals"] == m["home_goals"]: dr += 1
        n = len(h2h) or 1

        elo_h = elo_map.get(home_team.lower(), 1500.0)
        elo_a = elo_map.get(away_team.lower(), 1500.0)

        comp_oh = [1.0 if competition == c else 0.0 for c in self.COMPETITIONS]

        feats = [
            hs["form_gf"], as_["form_gf"],
            hs["form_pts"], as_["form_pts"],
            hs["avg_gf"],  as_["avg_gf"],
            hs["avg_ga"],  as_["avg_ga"],
            hs["win_rate"],as_["win_rate"],
            float(elo_h - elo_a),
            0.0,   # home_advantage (neutral en WC)
            1.0,   # is_neutral
            hw / n, dr / n,
        ] + comp_oh

        return np.array(feats, dtype=np.float32)