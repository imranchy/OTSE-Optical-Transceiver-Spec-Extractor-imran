import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "gold_dataset" / "gold_candidates"

def parse_json_or_none(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    return json.loads(str(x))

def main():
    decisions = pd.read_csv(OUT / "final_scalar_resolutions.csv")
    base = pd.read_csv(OUT / "remaining_unresolved.csv")

    by_idx = {int(r["row_index"]): r for _, r in decisions.iterrows()}
    unresolved_after = []

    for _, r in base.iterrows():
        idx = int(r["row_index"])
        if idx not in by_idx:
            unresolved_after.append(r.to_dict())

    pd.DataFrame(unresolved_after).to_csv(OUT / "scalar_unresolved_after_final_review.csv", index=False)

    print(f"Final scalar decisions: {len(decisions)}")
    print(f"Remaining scalar review: {len(unresolved_after)}")
    print("Next stage: merge accepted/adjudicated fields into final gold specs and review application modes.")

if __name__ == "__main__":
    main()
