import json
import logging
import numpy as np
import pandas as pd
from config import MATCHES_CSV, ELO_CSV, MODEL_DIR, LOGS_DIR
from model import XGBoostPredictor, FeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "train.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MODEL_PATH  = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
REPORT_PATH = MODEL_DIR / "train_report.json"


def train():
    if not MATCHES_CSV.exists():
        raise FileNotFoundError(f"{MATCHES_CSV} not found. Run pipeline.py first.")

    matches = pd.read_csv(MATCHES_CSV, parse_dates=["date"])
    logger.info(f"Loaded {len(matches):,} matches")

    elo = pd.read_csv(ELO_CSV) if ELO_CSV.exists() else pd.DataFrame()
    if not elo.empty:
        logger.info(f"ELO loaded for {len(elo)} teams")
    else:
        logger.warning("ELO not found — elo_diff = 0")

    builder = FeatureBuilder()
    X, y, feature_names = builder.fit_transform(matches, elo if not elo.empty else None)

    if len(X) == 0:
        raise ValueError("Feature matrix is empty.")

    logger.info(f"Feature matrix: {X.shape}")

    dist = pd.Series(y).value_counts(normalize=True).sort_index()
    logger.info(f"Classes → home_win: {dist.get(0,0):.1%}, draw: {dist.get(1,0):.1%}, away_win: {dist.get(2,0):.1%}")

    # Split temporal — nunca aleatorio en series de partidos
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    predictor = XGBoostPredictor()
    logger.info("Training...")
    train_m = predictor.train(X_train, y_train, feature_names=feature_names)
    logger.info(f"Train: {train_m}")

    X_test_scaled = predictor.scaler.transform(X_test)
    test_m = predictor.evaluate(X_test_scaled, y_test)
    logger.info(f"Test:  {test_m}")

    predictor.save(str(MODEL_PATH), str(SCALER_PATH))
    logger.info(f"Saved → {MODEL_PATH}")

    report = {
        "n_samples":      int(len(X)),
        "n_features":     int(len(feature_names)),
        "feature_names":  feature_names,
        "class_dist":     {int(k): float(v) for k, v in dist.items()},
        "train_metrics":  train_m,
        "test_metrics":   test_m,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report, predictor


if __name__ == "__main__":
    report, _ = train()
    print(f"\nTest accuracy:  {report['test_metrics']['accuracy']:.3f}")
    auc = report['test_metrics'].get('auc_ovo')
    if auc:
        print(f"Test AUC (OvO): {auc:.3f}")