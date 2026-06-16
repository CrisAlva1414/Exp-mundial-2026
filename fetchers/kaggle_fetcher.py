import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from .base_fetcher import BaseFetcher


class KaggleFetcher(BaseFetcher):
    FILES = [
        "results.csv",
        "goalscorers.csv",
        "shootouts.csv",
        "former_names.csv",
    ]

    def __init__(self, data_dir: Path, logger):
        super().__init__("kaggle", data_dir, logger)

    def fetch(self) -> dict:
        try:
            dataset_path = Path(
                kagglehub.dataset_download(
                    "martj42/international-football-results-from-1872-to-2017"
                )
            )

            saved = []

            for filename in self.FILES:
                source = dataset_path / filename

                if not source.exists():
                    self.logger.warning(f"{filename} not found")
                    continue

                df = pd.read_csv(source)

                # Normalizar nombres de columnas
                df.columns = (
                    df.columns
                    .str.strip()
                    .str.lower()
                )

                # Ordenar cronológicamente si existe fecha
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date")
                    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

                name = filename.replace(".csv", "")
                output_path = (
                    self.data_dir /
                    f"kaggle_{name}_current.csv"
                )

                df.to_csv(output_path, index=False)

                self.logger.info(
                    f"[kaggle] Saved {len(df):,} rows -> {output_path.name}"
                )

                saved.append(output_path.name)

            return {
                "success": True,
                "data": saved
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": []
            }

    def save(self, data):
        # fetch() ya guarda los archivos
        return len(data)