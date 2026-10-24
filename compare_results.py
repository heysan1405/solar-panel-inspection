
import pandas as pd
from pathlib import Path

base = Path("PVEL_Detection_Repaired")

files = {
    "Original GhostLite": base / "ghostlite" / "results.csv",
    "Additional 50": base / "ghostlite_additional50" / "results.csv",
}

for name, path in files.items():
    print(f"\n{'=' * 50}")
    print(name)
    print(f"File: {path}")
    print(f"Exists: {path.exists()}")

    if not path.exists():
        continue

    df = pd.read_csv(path)

    columns = [
        col for col in df.columns
        if any(key in col.lower() for key in [
            "epoch", "precision", "recall", "map50"
        ])
    ]

    print("\nLast 5 epochs:")
    print(df[columns].tail(5).to_string(index=False))

    map_columns = [
        col for col in df.columns
        if "map50-95" in col.lower()
    ]

    if map_columns:
        col = map_columns[0]
        best_index = df[col].idxmax()
        print(f"\nBest {col}:")
        print(df.loc[best_index, columns].to_string())
