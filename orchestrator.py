import signal
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from schedule import Scheduler
from config import (
    LOGS_DIR, STATE_FILE, FETCH_SCHEDULE, DATA_DIR
)
from fetchers.football_data_fetcher import FootballDataFetcher
from fetchers.eloratings_fetcher import EloratingsFetcher
from fetchers.openfootball_fetcher import OpenFootballFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

stop_event = threading.Event()

def handle_signal(sig, frame):
    logger.warning("\n Stop signal recibido. Terminando limpiamente...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_runs": {}}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

class Orchestrator:
    def __init__(self):
        self.scheduler = Scheduler()
        self.state = load_state()
        self.fetchers = [
            FootballDataFetcher(DATA_DIR, logger),
            EloratingsFetcher(DATA_DIR, logger),
            OpenFootballFetcher(DATA_DIR, logger),
        ]
        
        # Registrar jobs
        for fetcher in self.fetchers:
            cron_expr = FETCH_SCHEDULE.get(fetcher.name)
            if cron_expr:
                self.scheduler.cron(cron_expr).do(self.run_fetcher, fetcher)
        
        logger.info(" Orchestrator inicializado")
        logger.info(f"   Fetchers registrados: {len(self.fetchers)}")
        for f in self.fetchers:
            logger.info(f"     - {f.name}")
    
    def run_fetcher(self, fetcher):
        if stop_event.is_set():
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Ejecutando: {fetcher.name}")
        logger.info(f"{'='*60}")
        
        result = fetcher.run()
        
        # Guardar en state
        self.state["last_runs"][fetcher.name] = result
        save_state(self.state)
    
    def run_all_once(self):
        logger.info("\n" + "="*60)
        logger.info("EJECUCIÓN MANUAL: Todos los fetchers")
        logger.info("="*60)
        
        for fetcher in self.fetchers:
            if stop_event.is_set():
                break
            self.run_fetcher(fetcher)
    
    def run_scheduler(self):
        logger.info("\n" + "="*60)
        logger.info("INICIANDO SCHEDULER (Ctrl+C para detener)")
        logger.info("="*60)
        
        while not stop_event.is_set():
            self.scheduler.run_pending()
            
            # Chequear cada 60s si hay algo que hacer
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                stop_event.set()
    
    def run_loop_interval(self, interval_minutes: int = 60):
        logger.info("\n" + "="*60)
        logger.info(f"INICIANDO LOOP SIMPLE ({interval_minutes}min entre ciclos)")
        logger.info("="*60)
        
        cycle = 0
        while not stop_event.is_set():
            cycle += 1
            logger.info(f"\n>>> CICLO #{cycle} [{datetime.now().strftime('%H:%M:%S')}]")
            
            for fetcher in self.fetchers:
                if stop_event.is_set():
                    break
                self.run_fetcher(fetcher)
            
            logger.info(f"💤 Esperando {interval_minutes}min...\n")
            stop_event.wait(timeout=interval_minutes * 60)

if __name__ == "__main__":
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description="Football Data Orchestrator")
    parser.add_argument("--mode", choices=["once", "interval", "cron"], default="interval",
                       help="Modo: once (única ejecución), interval (loop simple), cron (scheduler)")
    parser.add_argument("--interval", type=int, default=60, help="Minutos entre ciclos (solo si mode=interval)")
    args = parser.parse_args()
    
    orch = Orchestrator()
    
    try:
        if args.mode == "once":
            orch.run_all_once()
        elif args.mode == "interval":
            orch.run_loop_interval(args.interval)
        elif args.mode == "cron":
            orch.run_scheduler()
    
    finally:
        logger.info("\n Orchestrator detenido limpiamente")