#!/usr/bin/env python3
"""
Football World Cup Predictor - Data Ingestors

Usage:
  python run.py                    # run all fetchers
  python run.py --skip-kaggle      # skip kaggle (already loaded)
  python run.py --only elo         # only run eloratings scraper
  python run.py --only football    # only run football-data.org
  python run.py --only openfootball
  python run.py --only kaggle

Flow: kaggle → openfootball → eloratings → football_data
Each fetcher returns {success, new_records, timestamp}
"""

import sys
import argparse
import logging
import signal
from pathlib import Path
from datetime import datetime
from config import LOGS_DIR, DATA_DIR

from fetchers.football_data_fetcher import FootballDataFetcher
from fetchers.eloratings_fetcher import EloratingsFetcher
from fetchers.openfootball_fetcher import OpenFootballFetcher
from fetchers.kaggle_fetcher import KaggleFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

stop_event = False


def handle_signal(sig, frame):
    global stop_event
    logger.warning("\n⚡ Stop signal received. Exiting gracefully...")
    stop_event = True
    sys.exit(0)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def run_fetchers(skip_kaggle: bool = False, only: str = None) -> dict:
    results = {}

    fetchers_config = [
        ("kaggle", KaggleFetcher(DATA_DIR, logger), not skip_kaggle),
        ("openfootball", OpenFootballFetcher(DATA_DIR, logger), True),
        ("elo", EloratingsFetcher(DATA_DIR, logger), True),
        ("football", FootballDataFetcher(DATA_DIR, logger), True),
    ]

    for name, fetcher, should_run in fetchers_config:
        if only and name != only:
            continue
        if not should_run:
            logger.info(f"[{name}] Skipped")
            continue

        result = fetcher.run()
        results[name] = result

        if result["success"]:
            logger.info(f"[{name}] ✓ {result.get('new_records', 0)} new records")
        else:
            logger.error(f"[{name}] ✗ {result.get('error', 'Unknown error')}")

    return results


def print_summary(results: dict):
    print("\n" + "=" * 70)
    print(f"{'Fetcher':<15} {'Status':<10} {'Records':<10} {'Timestamp':<30}")
    print("=" * 70)

    total_records = 0
    for name, result in results.items():
        status = "✓" if result["success"] else "✗"
        records = result.get("new_records", 0)
        timestamp = result.get("timestamp", "-")[:19]

        print(f"{name:<15} {status:<10} {records:<10} {timestamp:<30}")
        if result["success"]:
            total_records += records

    print("=" * 70)
    print(f"Total new records: {total_records}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Football World Cup Predictor - Data Ingestors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip Kaggle fetcher (already loaded)",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["kaggle", "openfootball", "elo", "football"],
        help="Run only this fetcher",
    )

    args = parser.parse_args()

    logger.info(f"🚀 Starting data ingestors at {datetime.now().isoformat()}")
    logger.info(f"Skip Kaggle: {args.skip_kaggle}")
    if args.only:
        logger.info(f"Only running: {args.only}")

    try:
        results = run_fetchers(skip_kaggle=args.skip_kaggle, only=args.only)
        print_summary(results)
        logger.info(f"✅ Completed at {datetime.now().isoformat()}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print_summary(results if 'results' in locals() else {})
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
