from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

for csv_file in sorted(DATA_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(csv_file, nrows=3)

        print(f"\n=== {csv_file.name} ===")
        print(f"Columnas ({len(df.columns)}):")
        print(list(df.columns))
        print("\nPrimeras 3 filas:")
        print(df.to_string(index=False))

    except Exception as e:
        print(f"\nError leyendo {csv_file.name}: {e}")