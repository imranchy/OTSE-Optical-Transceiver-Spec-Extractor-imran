import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "gold_dataset"
REPORTS = GOLD / "reports"
OUT = GOLD / "gold_candidates"

def main():
    decisions = pd.read_csv(OUT / "semantic_resolutions.csv")
    unresolved = pd.read_csv(OUT / "unresolved.csv")

    if "row_index" not in decisions.columns:
        raise RuntimeError("semantic_resolutions.csv is missing row_index")

    resolved_ids = set(int(x) for x in decisions["row_index"].tolist())
    unresolved = unresolved.reset_index().rename(columns={"index":"row_index"})
    remaining = unresolved[~unresolved["row_index"].isin(resolved_ids)].copy()

    decisions.to_csv(OUT / "semantic_resolutions_applied.csv", index=False)
    remaining.to_csv(OUT / "remaining_unresolved.csv", index=False)

    print(f"Applied semantic decisions: {len(decisions)}")
    print(f"Remaining unresolved:      {len(remaining)}")
    print(f"Wrote: {OUT / 'semantic_resolutions_applied.csv'}")
    print(f"Wrote: {OUT / 'remaining_unresolved.csv'}")

if __name__ == "__main__":
    main()
