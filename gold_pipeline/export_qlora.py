from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import Counter
from typing import Any

from pipeline.atom_extractor import FIELD_GROUPS, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _extract_headings, _schema_for_fields
from transceiver_models import ApplicationModeSet, TransceiverSpecs


DEFAULT_GOLD = ROOT / "gold_dataset"
METADATA_FIELDS = {"notes", "suggestions", "provenance_datasheet_sections"}


def compact_schema_for_modes() -> dict[str, Any]:
    # Matches the current mode schema while avoiding a hard dependency on a private helper.
    schema = ApplicationModeSet.model_json_schema()
    schema.pop("title", None)
    return schema


def mode_constants():
    # Import at execution time so export always follows the user's current pipeline branch.
    from pipeline.mode_linker import MODE_SYSTEM_PROMPT, MODE_PROMPT_TEMPLATE, separate_atoms
    return MODE_SYSTEM_PROMPT, MODE_PROMPT_TEMPLATE, separate_atoms


def json_target(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_record(gold_dir: Path, doc: str) -> dict[str, Any]:
    return json.loads((gold_dir / "final_gold" / f"{doc}.json").read_text(encoding="utf-8"))


def source_text(gold_dir: Path, doc: str) -> str:
    return (gold_dir / "benchmark_runs" / doc / "source" / "content.md").read_text(encoding="utf-8", errors="replace")


def atomic_examples(gold_dir: Path, doc: str, split: str) -> list[dict[str, Any]]:
    rec = read_record(gold_dir, doc)
    specs = rec["specs"]
    content = source_text(gold_dir, doc)
    headings = _extract_headings(content)
    examples: list[dict[str, Any]] = []

    for group_name, original_fields in FIELD_GROUPS:
        # Metadata/provenance was intentionally not made part of the verified gold target.
        fields = tuple(f for f in original_fields if f not in METADATA_FIELDS)
        prompt = USER_PROMPT_TEMPLATE.format(
            schema=_schema_for_fields(fields),
            headings=headings,
            content=content,
        )
        target = {f: specs.get(f) for f in fields}
        # Validate by overlaying the target on model defaults.
        full = TransceiverSpecs().model_dump(mode="json")
        full.update(target)
        TransceiverSpecs.model_validate(full)

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json_target(target)},
            ],
            "task": "atomic_extraction",
            "field_group": group_name,
            "document_id": doc,
            "split": split,
            "dataset_version": "otse-gold-pilot-v1",
        })
    return examples


def mode_example(gold_dir: Path, doc: str, split: str) -> dict[str, Any] | None:
    rec = read_record(gold_dir, doc)
    if not rec.get("modes_reviewed") or not rec.get("modes"):
        return None

    MODE_SYSTEM_PROMPT, MODE_PROMPT_TEMPLATE, separate_atoms = mode_constants()
    specs = rec["specs"]
    _, general = separate_atoms(specs)
    content = source_text(gold_dir, doc)
    if len(content) > 80_000:
        content = content[:80_000] + "\n\n[...truncated...]"
    target = {"modes": rec["modes"]}
    ApplicationModeSet.model_validate(target)

    prompt = MODE_PROMPT_TEMPLATE.format(
        schema=json.dumps(compact_schema_for_modes(), indent=2),
        general=json.dumps(general, indent=2),
        content=content,
    )
    return {
        "messages": [
            {"role": "system", "content": MODE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json_target(target)},
        ],
        "task": "mode_extraction",
        "document_id": doc,
        "split": split,
        "dataset_version": "otse-gold-pilot-v1",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_jsonl(path: Path) -> Counter:
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            msgs = row.get("messages")
            if not isinstance(msgs, list) or [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
                raise ValueError(f"Bad messages format at {path}:{line_no}")
            assistant = json.loads(msgs[-1]["content"])
            if row["task"] == "mode_extraction":
                ApplicationModeSet.model_validate(assistant)
            elif row["task"] == "atomic_extraction":
                full = TransceiverSpecs().model_dump(mode="json")
                full.update(assistant)
                TransceiverSpecs.model_validate(full)
            else:
                raise ValueError(f"Unknown task at {path}:{line_no}")
            counts[row["task"]] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Export reviewed OTSE gold records to Qwen/TRL messages JSONL.")
    ap.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    args = ap.parse_args()

    gold = args.gold_dir
    manifest = json.loads((gold / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    out = gold / "qlora"
    out.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    task_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for split in ("train", "validation"):
        for doc in manifest[split]:
            atoms = atomic_examples(gold, doc, split)
            all_rows[split].extend(atoms)
            task_rows.setdefault((split, "atomic"), []).extend(atoms)
            mode = mode_example(gold, doc, split)
            if mode is not None:
                all_rows[split].append(mode)
                task_rows.setdefault((split, "modes"), []).append(mode)

    write_jsonl(out / "train.jsonl", all_rows["train"])
    write_jsonl(out / "validation.jsonl", all_rows["validation"])
    for (split, task), rows in task_rows.items():
        write_jsonl(out / f"{split}_{task}.jsonl", rows)

    train_counts = validate_jsonl(out / "train.jsonl")
    val_counts = validate_jsonl(out / "validation.jsonl")

    def approx_tokens(rows: list[dict[str, Any]]) -> int:
        # Conservative dependency-free estimate for planning only; training uses the real tokenizer later.
        chars = sum(sum(len(m["content"]) for m in r["messages"]) for r in rows)
        return (chars + 3) // 4

    summary = {
        "dataset_version": "otse-gold-pilot-v1",
        "train_documents": len(manifest["train"]),
        "validation_documents": len(manifest["validation"]),
        "heldout_pdfs": len(manifest.get("heldout_pdfs", [])),
        "train_examples": len(all_rows["train"]),
        "validation_examples": len(all_rows["validation"]),
        "train_task_counts": dict(train_counts),
        "validation_task_counts": dict(val_counts),
        "train_approx_tokens_chars_over_4": approx_tokens(all_rows["train"]),
        "validation_approx_tokens_chars_over_4": approx_tokens(all_rows["validation"]),
        "metadata_fields_excluded_from_atomic_sft": sorted(METADATA_FIELDS),
        "negative_mode_examples_exported": 0,
        "notes": [
            "Atomic examples use the current local OTSE system prompt, field groups, schema snippets, and parsed Markdown.",
            "Mode examples are exported only for documents whose mode rows were explicitly reviewed.",
            "No held-out PDF is read by this exporter.",
            "The character/4 token estimate is only a planning heuristic; use the Qwen tokenizer before training.",
        ],
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote QLoRA files to {out}")


if __name__ == "__main__":
    main()
