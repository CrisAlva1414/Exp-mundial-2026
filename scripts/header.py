from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

for csv_file in sorted(DATA_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(csv_file, nrows=5)

        print(f"\n=== {csv_file.name} ===")
        print(f"Rows sample: {len(df)}")
        print(f"Columns ({len(df.columns)}):")
        print(list(df.columns))

    except Exception as e:
        print(f"Error en {csv_file.name}: {e}")