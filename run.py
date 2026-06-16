#!/usr/bin/env python3
"""
Football World Cup 2026 Predictor

Usage:
  python run.py                    # flujo completo
  python run.py --skip-fetch       # salta fetchers si los CSV ya existen
  python run.py --only-predict     # solo predice (modelo ya entrenado)
  python run.py --only elo         # solo un fetcher
  python run.py --force-retrain    # fuerza reentrenamiento aunque exista modelo
"""

import sys
import argparse
import logging
import signal
from pathlib import Path
from datetime import datetime

from config import LOGS_DIR, DATA_DIR, MATCHES_CSV, ELO_CSV, MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "run.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MODEL_PATH  = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# CSVs que deben existir para saltar los fetchers
REQUIRED_CSVS = [
    DATA_DIR / "kaggle_results_current.csv",
    DATA_DIR / "elo_current.csv",
    DATA_DIR / "openfootball_current.csv",
]


def handle_signal(sig, frame):
    logger.warning("Stop signal received. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def csvs_exist() -> bool:
    missing = [p for p in REQUIRED_CSVS if not p.exists()]
    if missing:
        logger.info(f"Missing CSVs: {[p.name for p in missing]}")
        return False
    logger.info("All source CSVs present — skipping fetchers")
    return True


def run_fetchers(only: str = None) -> dict:
    from fetchers.football_data_fetcher import FootballDataFetcher
    from fetchers.eloratings_fetcher    import EloratingsFetcher
    from fetchers.openfootball_fetcher  import OpenFootballFetcher
    from fetchers.kaggle_fetcher        import KaggleFetcher

    fetchers = [
        ("kaggle",       KaggleFetcher(DATA_DIR, logger)),
        ("openfootball", OpenFootballFetcher(DATA_DIR, logger)),
        ("elo",          EloratingsFetcher(DATA_DIR, logger)),
        ("football",     FootballDataFetcher(DATA_DIR, logger)),
    ]

    results = {}
    for name, fetcher in fetchers:
        if only and name != only:
            continue
        result = fetcher.run()
        results[name] = result
        if result["success"]:
            logger.info(f"[{name}] ✓ {result.get('new_records', 0)} records")
        else:
            logger.error(f"[{name}] ✗ {result.get('error', 'unknown')}")
    return results


def run_pipeline():
    from pipeline import run_pipeline as _pipeline
    logger.info("── Pipeline: merging sources → matches.csv ──")
    df = _pipeline()
    if df.empty:
        raise RuntimeError("Pipeline produced no data.")
    logger.info(f"Pipeline done: {len(df):,} matches")
    return df


def run_train():
    from train import train
    logger.info("── Training XGBoost model ──")
    report, predictor = train()
    logger.info(f"Train done — test accuracy: {report['test_metrics']['accuracy']:.3f}")
    return report


def run_predict():
    from wc2026_predictor import run_predictions
    logger.info("── Predicting World Cup 2026 ──")
    results = run_predictions()
    logger.info(f"Predicted {len(results['predictions'])} matches")
    return results


def print_summary(steps: list):
    print(f"\n{'='*50}")
    print(f"{'Step':<20} {'Status'}")
    print(f"{'='*50}")
    for step, ok, detail in steps:
        mark = "✓" if ok else "✗"
        print(f"{step:<20} {mark}  {detail}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-fetch",    action="store_true", help="Saltar fetchers si los CSV existen")
    parser.add_argument("--only-predict",  action="store_true", help="Solo predecir (modelo ya entrenado)")
    parser.add_argument("--force-retrain", action="store_true", help="Forzar reentrenamiento")
    parser.add_argument("--only",          type=str, choices=["kaggle", "openfootball", "elo", "football"])
    args = parser.parse_args()

    logger.info(f"🚀 Starting at {datetime.now().isoformat()}")
    steps = []

    # ── FETCH ────────────────────────────────────────────────────
    if not args.only_predict:
        should_fetch = not (args.skip_fetch or csvs_exist())

        if should_fetch:
            try:
                results = run_fetchers(only=args.only)
                ok = all(r["success"] for r in results.values())
                steps.append(("fetch", ok, f"{sum(r.get('new_records',0) for r in results.values())} new records"))
            except Exception as e:
                logger.error(f"Fetch failed: {e}")
                steps.append(("fetch", False, str(e)))
                sys.exit(1)
        else:
            steps.append(("fetch", True, "skipped (CSVs present)"))

        # ── PIPELINE ─────────────────────────────────────────────
        if not args.only:
            try:
                df = run_pipeline()
                steps.append(("pipeline", True, f"{len(df):,} matches → matches.csv"))
            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                steps.append(("pipeline", False, str(e)))
                sys.exit(1)

        # ── TRAIN ────────────────────────────────────────────────
        if not args.only:
            model_exists = MODEL_PATH.exists() and SCALER_PATH.exists()
            if not model_exists or args.force_retrain:
                try:
                    report = run_train()
                    acc = report["test_metrics"]["accuracy"]
                    steps.append(("train", True, f"test accuracy {acc:.3f}"))
                except Exception as e:
                    logger.error(f"Train failed: {e}")
                    steps.append(("train", False, str(e)))
                    sys.exit(1)
            else:
                logger.info("Model already exists — skipping train (use --force-retrain to override)")
                steps.append(("train", True, "skipped (model present)"))

    # ── PREDICT ──────────────────────────────────────────────────
    if not args.only:
        try:
            run_predict()
            steps.append(("predict", True, "WC 2026 predictions done"))
        except Exception as e:
            logger.error(f"Predict failed: {e}", exc_info=True)
            steps.append(("predict", False, str(e)))

    print_summary(steps)
    logger.info(f"✅ Done at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()