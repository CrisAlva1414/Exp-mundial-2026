import time
import threading
import signal
import sys
from ingestor import FootballIngestor
from config import COMPETITIONS

stop_event = threading.Event()

def handle_signal(sig, frame):
    print("\n⚡ Stop signal recibido. Terminando limpiamente...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_signal)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_signal)  # kill

def run():
    ingestor = FootballIngestor()
    
    # Seasons a poblar (ajusta según lo que cubre tu plan)
    seasons = [2020, 2021, 2022, 2023, 2024]
    competitions = list(COMPETITIONS.keys())
    
    cycle = 0
    while not stop_event.is_set():
        cycle += 1
        print(f"\n=== Ciclo #{cycle} — {time.strftime('%H:%M:%S')} ===")
        
        for comp in competitions:
            if stop_event.is_set():
                break
            print(f"[{comp}]")
            ingestor.ingest_competition(comp, seasons)
        
        # Esperar 1 hora antes del próximo ciclo (o hasta stop)
        print("\n💤 Esperando 1h para el próximo ciclo... (Ctrl+C para detener)")
        stop_event.wait(timeout=3600)
    
    print("Detenido limpiamente.")

if __name__ == "__main__":
    run()