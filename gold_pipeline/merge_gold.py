from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import defaultdict
from typing import Any

import pandas as pd

from transceiver_models import ApplicationModeSet, TransceiverSpecs


DEFAULT_GOLD = ROOT / "gold_dataset"


def parse_json_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s.lower() == "nan":
        return None
    return json.loads(s)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def valid_documents(runs_dir: Path) -> list[str]:
    docs: list[str] = []
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        content = p / "source" / "content.md"
        summary = p / "source" / "parsed_summary.json"
        if not content.exists() or not summary.exists():
            continue
        # Exclude stale/failed smoke-test parses; the real corpus is native-text.
        if len(content.read_text(encoding="utf-8", errors="replace")) < 1000:
            continue
        docs.append(p.name)
    return docs


def apply_decision(
    specs_by_doc: dict[str, dict[str, Any]],
    audit_by_doc: dict[str, dict[str, Any]],
    *,
    doc: str,
    field: str,
    value: Any,
    stage: str,
    decision: str,
    rationale: str = "",
) -> None:
    if doc not in specs_by_doc:
        return
    if field not in TransceiverSpecs.model_fields:
        raise ValueError(f"Unknown TransceiverSpecs field {field!r} in {stage} for {doc}")
    specs_by_doc[doc][field] = value
    audit_by_doc[doc][field] = {
        "stage": stage,
        "decision": decision,
        "value": value,
        "rationale": rationale,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge all adjudicated OTSE labels into final per-document gold records.")
    ap.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    args = ap.parse_args()

    gold = args.gold_dir
    runs = gold / "benchmark_runs"
    cand = gold / "gold_candidates"
    out = gold / "final_gold"
    out.mkdir(parents=True, exist_ok=True)

    docs = valid_documents(runs)
    if not docs:
        raise SystemExit(f"No valid benchmark documents found under {runs}")

    defaults = TransceiverSpecs().model_dump(mode="json")
    specs_by_doc = {doc: dict(defaults) for doc in docs}
    audit_by_doc: dict[str, dict[str, Any]] = {doc: {} for doc in docs}

    # 1) Initial conservative adjudication (includes model agreement + safe promotions).
    accepted = read_csv(cand / "accepted_fields.csv")
    for _, r in accepted.iterrows():
        apply_decision(
            specs_by_doc, audit_by_doc,
            doc=str(r["document_id"]), field=str(r["field"]),
            value=parse_json_cell(r.get("gold_value")),
            stage="initial_adjudication", decision=str(r.get("decision", "ACCEPT")),
            rationale=str(r.get("rule", "")),
        )

    # 2) Second semantic pass. These intentionally override earlier candidates.
    semantic = read_csv(cand / "semantic_resolutions_applied.csv")
    for _, r in semantic.iterrows():
        apply_decision(
            specs_by_doc, audit_by_doc,
            doc=str(r["document_id"]), field=str(r["field"]),
            value=parse_json_cell(r.get("gold_value")),
            stage="semantic_review", decision=str(r.get("decision", "ACCEPT_SEMANTIC_REVIEW")),
            rationale=str(r.get("rationale", "")),
        )

    # 3) Final 34 scalar decisions. LEAVE_NULL is a deliberate gold decision.
    scalar = read_csv(cand / "final_scalar_resolutions.csv")
    for _, r in scalar.iterrows():
        decision = str(r.get("final_decision", ""))
        value = None if decision == "LEAVE_NULL" else parse_json_cell(r.get("final_gold_value"))
        apply_decision(
            specs_by_doc, audit_by_doc,
            doc=str(r["document_id"]), field=str(r["field"]), value=value,
            stage="final_scalar_review", decision=decision,
            rationale=str(r.get("final_rationale", "")),
        )

    # 4) Final reviewed application modes, grouped by document.
    mode_df = read_csv(cand / "final_mode_resolutions.csv")
    modes_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, r in mode_df.sort_values(["document_id", "mode_index"]).iterrows():
        doc = str(r["document_id"])
        mode = parse_json_cell(r["gold_mode_json"])
        modes_by_doc[doc].append(mode)

    manifest_rows: list[dict[str, Any]] = []
    for doc in docs:
        specs = TransceiverSpecs.model_validate(specs_by_doc[doc]).model_dump(mode="json")
        modes = modes_by_doc.get(doc, [])
        if modes:
            modes = ApplicationModeSet.model_validate({"modes": modes}).model_dump(mode="json")["modes"]

        summary = json.loads((runs / doc / "source" / "parsed_summary.json").read_text(encoding="utf-8"))
        record = {
            "dataset_version": "otse-gold-pilot-v1",
            "document_id": doc,
            "source_file": summary.get("source_file"),
            "source_sha256": summary.get("source_sha256"),
            "page_count": summary.get("page_count"),
            "table_count": summary.get("table_count"),
            "specs": specs,
            "modes_reviewed": bool(modes),
            "modes": modes,
            "field_audit": audit_by_doc[doc],
        }
        (out / f"{doc}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest_rows.append({
            "document_id": doc,
            "source_file": record["source_file"],
            "source_sha256": record["source_sha256"],
            "reviewed_field_count": len(audit_by_doc[doc]),
            "mode_count": len(modes),
            "modes_reviewed": bool(modes),
        })

    pd.DataFrame(manifest_rows).to_csv(out / "final_gold_manifest.csv", index=False)
    summary = {
        "dataset_version": "otse-gold-pilot-v1",
        "documents": len(docs),
        "total_reviewed_field_decisions": sum(len(x) for x in audit_by_doc.values()),
        "documents_with_reviewed_modes": sum(bool(modes_by_doc.get(d)) for d in docs),
        "reviewed_modes": sum(len(modes_by_doc.get(d, [])) for d in docs),
    }
    (out / "final_gold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote final gold records to {out}")


if __name__ == "__main__":
    main()
