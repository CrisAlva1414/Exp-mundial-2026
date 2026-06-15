import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

class BaseFetcher(ABC):
    
    def __init__(self, name: str, data_dir: Path, logger: logging.Logger):
        self.name = name
        self.data_dir = data_dir
        self.logger = logger
        self.last_call = 0
        self.min_interval = 0
    
    def rate_limit(self):
        if self.min_interval > 0:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    @abstractmethod
    def fetch(self) -> dict:
        pass
    
    @abstractmethod
    def save(self, data: list) -> int:
        pass
    
    def run(self) -> dict:
        try:
            self.logger.info(f"[{self.name}] Iniciando fetch...")
            result = self.fetch()
            
            if not result["success"]:
                self.logger.error(f"[{self.name}] Fetch falló: {result['error']}")
                return {
                    "name": self.name,
                    "success": False,
                    "error": result["error"],
                    "timestamp": datetime.now().isoformat(),
                }
            
            new_count = self.save(result["data"])
            self.logger.info(f"[{self.name}] ✓ {new_count} registros nuevos")
            
            return {
                "name": self.name,
                "success": True,
                "new_records": new_count,
                "timestamp": datetime.now().isoformat(),
            }
        
        except Exception as e:
            self.logger.error(f"[{self.name}] Excepción: {e}", exc_info=True)
            return {
                "name": self.name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }