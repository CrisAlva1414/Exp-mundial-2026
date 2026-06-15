# Football World Cup Predictor 2026 — Setup Guide

## 1. Virtual Environment

If your system supports it, create a dedicated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 2.0
- numpy >= 1.24
- requests >= 2.31
- beautifulsoup4 >= 4.12
- xgboost >= 2.0
- scikit-learn >= 1.3
- shap >= 0.44
- kaggle >= 1.6

## 3. Set API Keys

Set the football-data.org API key:

```bash
export FD_API_KEY="your_api_key_here"
```

Get a free API key at: https://www.football-data.org/

## 4. Optional: Kaggle Dataset

For additional historical data, download from Kaggle:

```bash
# Requires kaggle CLI configured with credentials
kaggle datasets download -d martj42/international-football-results-from-1872-to-2017
unzip international-football-results-from-1872-to-2017.zip -d data/raw/
```

Or download manually and extract `results.csv` to `data/raw/results.csv`

## 5. First Run

Run all fetchers:

```bash
python run.py
```

This will:
1. Load Kaggle data (if available)
2. Fetch World Cup matches from OpenFootball (2014, 2018, 2022, 2026)
3. Scrape current ELO ratings from eloratings.net
4. Fetch league matches from football-data.org (requires API key)

Result files created:
- `data/matches.csv` — Combined matches from all sources
- `data/elo_current.csv` — Current ELO ratings
- `state.json` — Checkpoint state for incremental fetching

## 6. Verify Installation

```bash
python -c "
import pandas, numpy, requests, bs4, xgboost, sklearn, shap
from config import COMPETITIONS, MATCHES_CSV, ELO_CSV
from storage import load_matches, load_elo, load_state
from fetchers.base_fetcher import BaseFetcher
from fetchers.openfootball_fetcher import OpenFootballFetcher
from fetchers.eloratings_fetcher import EloratingsFetcher
from fetchers.kaggle_fetcher import KaggleFetcher
from fetchers.football_data_fetcher import FootballDataFetcher
from model import XGBoostPredictor, FeatureBuilder
from predictor import PoissonSimulator, FootballPredictor
print('✅ All imports OK')
"
```

## 7. Run Individual Fetchers

```bash
# Only Kaggle
python run.py --only kaggle

# Only OpenFootball (no API key needed)
python run.py --only openfootball

# Only ELO ratings scraper (no API key needed)
python run.py --only elo

# Only football-data.org (requires API key)
python run.py --only football

# Skip Kaggle (if already loaded)
python run.py --skip-kaggle
```

## Directory Structure

```
football_predictor/
├── .venv/                          # Virtual environment
├── config.py                       # Configuration (API keys, paths)
├── run.py                          # Main entry point
├── model.py                        # XGBoost model + feature builder
├── predictor.py                    # Poisson simulator + predictor
├── storage.py                      # CSV helpers
├── requirements.txt                # Python dependencies
│
├── fetchers/
│   ├── __init__.py
│   ├── base_fetcher.py             # Abstract base class
│   ├── football_data_fetcher.py    # football-data.org API
│   ├── eloratings_fetcher.py       # eloratings.net scraper
│   ├── openfootball_fetcher.py     # OpenFootball JSON fetcher
│   └── kaggle_fetcher.py           # Kaggle CSV reader
│
├── data/
│   ├── raw/                        # Raw data from sources
│   │   └── results.csv             # (manual Kaggle download)
│   ├── matches.csv                 # Combined matches (generated)
│   └── elo_current.csv             # Current ELO ratings (generated)
│
├── logs/
│   └── orchestrator.log            # Fetch logs
│
└── state.json                      # Checkpoint state
```

## Next Steps

After successful data ingestión:

1. **Train the model** (future: implement in separate script)
2. **Make predictions** (future: implement CLI interface)
3. **Deploy** (future: integrate with n8n/alerts)

See `README.md` for full architecture documentation.
