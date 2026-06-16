import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from datetime import datetime

from config import MATCHES_CSV, ELO_CSV, MODEL_DIR, WC2026_JSON
from model import XGBoostPredictor, FeatureBuilder

MODEL_PATH  = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Normalización de nombres: worldcup.json → nombres canónicos en el historial
NAME_MAP = {
    "Czech Republic":       "Czech Republic",
    "South Korea":          "South Korea",
    "Ivory Coast":          "Ivory Coast",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "DR Congo":             "DR Congo",
    "Cape Verde":           "Cape Verde",
    "USA":                  "United States",
    "Curaçao":              "Curaçao",
    "New Zealand":          "New Zealand",
}

def _norm(name: str) -> str:
    return NAME_MAP.get(name, name)


def _is_real_team(name: str) -> bool:
    if not name:
        return False
    placeholders = {f"{i}{g}" for i in "12" for g in "ABCDEFGHIJKL"}
    if name in placeholders:
        return False
    if name.startswith(("W", "L")) and name[1:].isdigit():
        return False
    if name.startswith("3"):
        return False
    return True


class PoissonSimulator:

    def estimate_lambdas(self, p_home: float, p_draw: float, p_away: float) -> tuple:
        # Media de goles en torneos internacionales ~2.5 total
        base = 1.25
        diff = (p_home - p_away) * 0.8
        lh = max(base + diff, 0.15)
        la = max(base - diff, 0.15)
        return lh, la

    def simulate(self, lh: float, la: float, n: int = 50_000) -> dict:
        hg = np.random.poisson(lh, n)
        ag = np.random.poisson(la, n)
        outcomes = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
        counts   = np.bincount(outcomes, minlength=3)
        p_hw, p_d, p_aw = counts / n

        score_dist  = Counter(zip(hg.tolist(), ag.tolist()))
        top_scores  = {f"{h}-{a}": round(v / n, 4) for (h, a), v in score_dist.most_common(8)}
        most_likely = score_dist.most_common(1)[0][0]

        return {
            "prob_home": round(float(p_hw), 4),
            "prob_draw": round(float(p_d),  4),
            "prob_away": round(float(p_aw), 4),
            "most_likely": f"{most_likely[0]}-{most_likely[1]}",
            "top_scores":  top_scores,
            "lambda_home": round(lh, 3),
            "lambda_away": round(la, 3),
        }


class WC2026Predictor:

    def __init__(self):
        self.xgb      = XGBoostPredictor.load(str(MODEL_PATH), str(SCALER_PATH))
        self.poisson  = PoissonSimulator()
        self.builder  = FeatureBuilder()
        self.history  = self._load_history()
        self.elo_map  = self._load_elo()
        self.matches  = self._load_wc_matches()

    def _load_history(self) -> pd.DataFrame:
        df = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _load_elo(self) -> dict:
        if not ELO_CSV.exists():
            return {}
        elo = pd.read_csv(ELO_CSV)
        return dict(zip(elo["team"].str.lower(), elo["elo_rating"].astype(float)))

    def _load_wc_matches(self) -> list:
        with open(WC2026_JSON) as f:
            data = json.load(f)

        matches = []
        for m in data["matches"]:
            t1 = m.get("team1", "")
            t2 = m.get("team2", "")
            if not _is_real_team(t1) or not _is_real_team(t2):
                continue
            matches.append({
                "date":  m.get("date", ""),
                "home":  t1,
                "away":  t2,
                "round": m.get("round", ""),
                "group": m.get("group", ""),
                "venue": m.get("ground", ""),
                "score": m.get("score", {}).get("ft"),  # None si no se jugó
            })
        return matches

    def predict_match(self, home: str, away: str, date: str) -> dict:
        h_norm = _norm(home)
        a_norm = _norm(away)
        ref    = pd.to_datetime(date)

        # Historia hasta antes del partido
        history = self.history[self.history["date"] < ref]

        X = self.builder.transform_single(
            h_norm, a_norm, "WC", ref, history, self.elo_map
        ).reshape(1, -1)

        probs = self.xgb.predict_proba(X)[0]
        p_hw, p_d, p_aw = float(probs[0]), float(probs[1]), float(probs[2])

        lh, la = self.poisson.estimate_lambdas(p_hw, p_d, p_aw)
        sim    = self.poisson.simulate(lh, la)

        elo_h = self.elo_map.get(h_norm.lower())
        elo_a = self.elo_map.get(a_norm.lower())

        return {
            "home":      home,
            "away":      away,
            "home_norm": h_norm,
            "away_norm": a_norm,
            "xgb": {
                "home_win": round(p_hw, 4),
                "draw":     round(p_d,  4),
                "away_win": round(p_aw, 4),
            },
            "simulation": sim,
            "elo": {
                "home": int(elo_h) if elo_h else None,
                "away": int(elo_a) if elo_a else None,
                "diff": int(elo_h - elo_a) if (elo_h and elo_a) else None,
            },
        }

    def predict_all(self) -> dict:
        played, pending = [], []

        for m in self.matches:
            if m["score"] and len(m["score"]) == 2:
                played.append(m)
            else:
                pending.append(m)

        print(f"\n{'='*60}")
        print(f"  FIFA WORLD CUP 2026 — PREDICCIONES COMPLETAS")
        print(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        print(f"  Partidos ya jugados:    {len(played)}")
        print(f"  Partidos a predecir:    {len(pending)}")
        print(f"{'='*60}\n")

        predictions = []

        # Agrupar por round
        current_round = None
        for m in pending:
            if m["round"] != current_round:
                current_round = m["round"]
                label = m.get("group", current_round)
                print(f"\n── {current_round} {'(' + label + ')' if label and label != current_round else ''} ──")

            pred = self.predict_match(m["home"], m["away"], m["date"])
            pred["date"]  = m["date"]
            pred["round"] = m["round"]
            pred["group"] = m.get("group", "")
            pred["venue"] = m.get("venue", "")
            predictions.append(pred)

            # Determinar favorito
            p = pred["xgb"]
            if p["home_win"] >= p["away_win"] and p["home_win"] >= p["draw"]:
                fav = f"→ {m['home']} ({p['home_win']:.0%})"
            elif p["away_win"] >= p["home_win"] and p["away_win"] >= p["draw"]:
                fav = f"→ {m['away']} ({p['away_win']:.0%})"
            else:
                fav = f"→ Empate ({p['draw']:.0%})"

            elo_str = ""
            if pred["elo"]["diff"] is not None:
                elo_str = f" | ELO diff: {pred['elo']['diff']:+d}"

            print(f"  {m['date']}  {m['home']:<22} vs  {m['away']:<22}")
            print(f"           Local {p['home_win']:.0%}  Empate {p['draw']:.0%}  Visita {p['away_win']:.0%}  {fav}{elo_str}")
            print(f"           Score más probable: {pred['simulation']['most_likely']}  |  λ({pred['simulation']['lambda_home']:.2f} - {pred['simulation']['lambda_away']:.2f})")

        print(f"\n{'='*60}")
        print(f"  RESUMEN — FAVORITOS POR GRUPO")
        print(f"{'='*60}")

        # Favorito por grupo
        group_preds = [p for p in predictions if p["group"].startswith("Group")]
        groups = {}
        for p in group_preds:
            g = p["group"]
            if g not in groups:
                groups[g] = {}
            # Sumar puntos esperados por equipo
            for team, prob_key in [(p["home"], "home_win"), (p["away"], "away_win")]:
                if team not in groups[g]:
                    groups[g][team] = 0.0
                groups[g][team] += p["xgb"][prob_key] * 3 + p["xgb"]["draw"]

        for g in sorted(groups.keys()):
            ranking = sorted(groups[g].items(), key=lambda x: x[1], reverse=True)
            print(f"\n  {g}:")
            for i, (team, pts) in enumerate(ranking, 1):
                elo = self.elo_map.get(_norm(team).lower())
                elo_str = f" (ELO {int(elo)})" if elo else ""
                print(f"    {i}. {team:<22} {pts:.2f} pts esperados{elo_str}")

        return {
            "played":      played,
            "predictions": predictions,
            "generated_at": datetime.now().isoformat(),
        }


def run_predictions() -> dict:
    predictor = WC2026Predictor()
    results   = predictor.predict_all()
    return results


if __name__ == "__main__":
    run_predictions()