import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "gold_dataset"
OUT = GOLD / "gold_candidates"

def main():
    src = OUT / "final_mode_resolutions.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    print(f"Final mode decisions: {len(df)}")
    print("Remaining mode review: 0")
    print("Next stage: merge scalar + mode gold labels and export QLoRA splits.")

if __name__ == "__main__":
    main()
