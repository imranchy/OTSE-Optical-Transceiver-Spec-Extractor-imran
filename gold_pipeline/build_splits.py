from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json


DEFAULT_GOLD = ROOT / "gold_dataset"

# Fixed before QLoRA training. These represent three distinct validation regimes:
# 100G LR4 direct-detect/WDM, 400G SR8 parallel PAM4, and coherent 100G ZR.
VALIDATION_PREFIXES = (
    "02-coherent-ftlc1154sdpl-100g-lr4-10km-qsfp28-",
    "07-coherent-ftcd8613e2pcm-400g-sr8-qsfpdd-",
    "qsfp28-100g-zr-high-tx-coherent-optics-module-ds-",
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create the fixed document-level OTSE train/validation/heldout split.")
    ap.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    args = ap.parse_args()

    gold = args.gold_dir
    final_dir = gold / "final_gold"
    split_dir = gold / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for p in sorted(final_dir.glob("*.json")):
        if p.name.endswith("summary.json"):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if "document_id" in data and "specs" in data:
            records.append(data)

    if len(records) != 15:
        raise SystemExit(f"Expected 15 reviewed development documents, found {len(records)}. Refusing to create a different split silently.")

    docs = [r["document_id"] for r in records]
    validation = [d for d in docs if any(d.startswith(prefix) for prefix in VALIDATION_PREFIXES)]
    if len(validation) != 3:
        raise SystemExit(f"Expected exactly 3 fixed validation documents, resolved {len(validation)}: {validation}")
    train = [d for d in docs if d not in validation]

    heldout_dir = gold / "heldout_pdfs"
    heldout = sorted(p.name for p in heldout_dir.glob("*.pdf")) if heldout_dir.exists() else []

    # Leakage checks.
    if set(train) & set(validation):
        raise RuntimeError("Train/validation document overlap detected")
    sha_by_doc = {r["document_id"]: r.get("source_sha256") for r in records}
    train_sha = {sha_by_doc[d] for d in train if sha_by_doc[d]}
    val_sha = {sha_by_doc[d] for d in validation if sha_by_doc[d]}
    if train_sha & val_sha:
        raise RuntimeError("Train/validation duplicate source SHA256 detected")

    manifest = {
        "dataset_version": "otse-gold-pilot-v1",
        "split_policy": "fixed_document_level_split_frozen_before_qlora",
        "train": train,
        "validation": validation,
        "heldout_pdfs": heldout,
        "counts": {"train": len(train), "validation": len(validation), "heldout": len(heldout)},
        "validation_rationale": {
            validation[0]: "100G LR4 / LAN-WDM direct-detect validation case",
            validation[1]: "400G SR8 / parallel PAM4 validation case",
            validation[2]: "100G coherent ZR validation case with reviewed modes",
        },
        "rules": [
            "All examples derived from one PDF remain in the same split.",
            "Held-out PDFs are excluded from gold generation, prompt tuning, QLoRA training, and validation.",
            "The split must not be changed in response to QLoRA results.",
        ],
    }
    path = split_dir / "split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))
    print("Validation documents:")
    for d in validation:
        print(f"  - {d}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
