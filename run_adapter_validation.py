"""Run the trained Qwen3-4B OTSE LoRA on ONLY the frozen validation documents.

This script deliberately reuses:
  * the exact parsed Markdown saved by the original stock-model benchmark,
  * pipeline.atom_extractor.extract_atoms(),
  * pipeline.mode_linker.synthesize_modes(),
  * the current Pydantic schemas in transceiver_models.py.

It never scans heldout_pdfs/ and refuses any document not listed in the frozen
validation split manifest.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf_adapter_backend import HFAdapterEngine, make_openai_factory
from pipeline.pdf_parser import ParsedDatasheet, ParsedPage, ParsedTable
import pipeline.atom_extractor as atom_extractor
from pipeline.atom_extractor import extract_atoms
from pipeline.mode_linker import store_modes, synthesize_modes

LOGGER = logging.getLogger("otse.adapter_validation")
EMPTY = (None, "", [], {})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def populated_fields(data: dict[str, Any] | None) -> int:
    if not data:
        return 0
    return sum(v not in EMPTY for v in data.values())


def reconstruct_parsed(doc_dir: Path, source_pdf_dir: Path) -> ParsedDatasheet:
    """Rebuild ParsedDatasheet from frozen benchmark artifacts without re-parsing PDF."""
    source_dir = doc_dir / "source"
    markdown_path = source_dir / "content.md"
    summary_path = source_dir / "parsed_summary.json"
    tables_path = source_dir / "tables.json"
    if not markdown_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing frozen parsed artifacts under {source_dir}")

    markdown = markdown_path.read_text(encoding="utf-8")
    summary = load_json(summary_path, {}) or {}
    source_file = summary.get("source_file")
    if not source_file:
        raise RuntimeError(f"No source_file in {summary_path}")

    page_count = int(summary.get("page_count", 0))
    pages = [ParsedPage(index=i, text="") for i in range(page_count)]
    tables = []
    for item in load_json(tables_path, []) or []:
        tables.append(
            ParsedTable(
                page_index=int(item.get("page_index", 0)),
                bbox=tuple(item.get("bbox", [0, 0, 0, 0])),
                rows=item.get("rows", []) or [],
                label=item.get("label"),
            )
        )

    return ParsedDatasheet(
        source_path=source_pdf_dir / source_file,
        pages=pages,
        tables=tables,
        markdown=markdown,
    )


def validate_split(split_manifest: dict[str, Any], benchmark_root: Path) -> list[str]:
    validation = list(split_manifest.get("validation", []))
    heldout = set(split_manifest.get("heldout_pdfs", []))
    if len(validation) != 3:
        raise RuntimeError(f"Expected exactly 3 frozen validation documents, found {len(validation)}")
    if heldout & set(validation):
        raise RuntimeError("Held-out PDFs leaked into the validation list.")
    missing = [doc_id for doc_id in validation if not (benchmark_root / doc_id / "source" / "content.md").exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark Markdown for: " + ", ".join(missing))
    return validation


def run_document(
    *,
    doc_id: str,
    parsed: ParsedDatasheet,
    output_root: Path,
    engine: HFAdapterEngine,
    max_chars: int,
    atom_max_tokens: int,
    mode_max_tokens: int,
    force: bool,
) -> dict[str, Any]:
    out_dir = output_root / doc_id / "qwen3_4b_otse_qlora"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists() and not force:
        old = load_json(metrics_path, {}) or {}
        if old.get("status") in {"success", "partial"}:
            LOGGER.info("SKIP completed %s", doc_id)
            return old

    before_records = len(engine.records)
    started = time.perf_counter()
    metrics: dict[str, Any] = {
        "status": "running",
        "started_at_utc": utc_now(),
        "document_id": doc_id,
        "source_file": parsed.source_path.name,
        "model": engine.model_id,
        "base_model_id": engine.base_model_id,
        "adapter_dir": str(engine.adapter_dir),
        "temperature": 0.0,
        "context_limit": engine.context_limit,
        "max_chars": max_chars,
        "atom_max_tokens": atom_max_tokens,
        "mode_max_tokens": mode_max_tokens,
        "markdown_chars": len(parsed.markdown),
        "page_count": parsed.page_count,
        "table_count": len(parsed.tables),
    }
    json_write(metrics_path, metrics)

    try:
        atom_started = time.perf_counter()
        specs, raw_atoms = extract_atoms(
            parsed,
            model_id="qwen3-4b-otse-qlora",
            max_chars=max_chars,
            base_url="inprocess://hf-adapter",
            temperature=0.0,
            max_tokens=atom_max_tokens,
        )
        atom_seconds = time.perf_counter() - atom_started
        json_write(out_dir / "raw_llm.json", raw_atoms)

        specs_dict = None
        if specs is not None:
            specs_dict = specs.model_dump(mode="json")
            json_write(out_dir / "specs.json", specs_dict)

        metrics.update(
            atomic_seconds=round(atom_seconds, 4),
            atomic_validation_passed=specs is not None,
            populated_spec_fields=populated_fields(specs_dict or raw_atoms),
        )

        mode_seconds = None
        modes_validation_passed = None
        modes_count = 0
        if specs_dict is not None:
            mode_started = time.perf_counter()
            modes, multi, general = synthesize_modes(
                specs_dict,
                parsed,
                run_dir=out_dir,
                model_id="qwen3-4b-otse-qlora",
                base_url="inprocess://hf-adapter",
                temperature=0.0,
                max_tokens=mode_max_tokens,
            )
            mode_seconds = time.perf_counter() - mode_started
            modes_validation_passed = modes is not None
            if modes is not None:
                store_modes(out_dir, modes, multi, general)
                modes_count = len(modes.get("modes", []))
            else:
                json_write(out_dir / "mode_atoms.json", {"multi_valued": multi, "general": general})

        new_records = engine.records[before_records:]
        json_write(
            out_dir / "generation_records.json",
            [record.__dict__ for record in new_records],
        )
        metrics.update(
            mode_seconds=None if mode_seconds is None else round(mode_seconds, 4),
            modes_validation_passed=modes_validation_passed,
            modes_count=modes_count,
            generation_calls=len(new_records),
            input_tokens_total=sum(r.input_tokens for r in new_records),
            output_tokens_total=sum(r.output_tokens for r in new_records),
            total_seconds=round(time.perf_counter() - started, 4),
            finished_at_utc=utc_now(),
            status="success" if specs is not None else "partial",
        )
        json_write(metrics_path, metrics)
        return metrics

    except Exception as exc:
        metrics.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            total_seconds=round(time.perf_counter() - started, 4),
            finished_at_utc=utc_now(),
        )
        (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        json_write(metrics_path, metrics)
        LOGGER.exception("FAILED %s", doc_id)
        return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=ROOT / "training_outputs" / "qwen3-4b-thinking-2507-otse-qlora" / "adapter_final",
    )
    ap.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "gold_dataset" / "splits" / "split_manifest.json",
    )
    ap.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "gold_dataset" / "benchmark_runs",
    )
    ap.add_argument(
        "--source-pdf-dir",
        type=Path,
        default=ROOT / "gold_dataset" / "source_pdfs",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "gold_dataset" / "adapter_validation_runs",
    )
    ap.add_argument("--context-limit", type=int, default=8192)
    ap.add_argument("--max-chars", type=int, default=120_000)
    ap.add_argument("--atom-max-tokens", type=int, default=8192)
    ap.add_argument("--mode-max-tokens", type=int, default=4096)
    ap.add_argument("--max-new-tokens-cap", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    split_manifest = load_json(args.split_manifest.resolve())
    if not isinstance(split_manifest, dict):
        raise SystemExit(f"Could not read split manifest: {args.split_manifest}")
    validation_ids = validate_split(split_manifest, args.benchmark_root.resolve())

    LOGGER.info("Frozen validation documents (%d): %s", len(validation_ids), ", ".join(validation_ids))
    LOGGER.info("Held-out PDFs are NOT accessed by this script: %s", split_manifest.get("heldout_pdfs", []))

    engine = HFAdapterEngine(
        base_model_id=args.base_model,
        adapter_dir=args.adapter_dir.resolve(),
        context_limit=args.context_limit,
        max_new_tokens_cap=args.max_new_tokens_cap,
    )

    # Route the production extraction code through our in-process adapter backend.
    atom_extractor.OpenAI = make_openai_factory(engine)

    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    rows = []
    for index, doc_id in enumerate(validation_ids, 1):
        LOGGER.info("[%d/%d] %s", index, len(validation_ids), doc_id)
        parsed = reconstruct_parsed(
            args.benchmark_root.resolve() / doc_id,
            args.source_pdf_dir.resolve(),
        )
        rows.append(
            run_document(
                doc_id=doc_id,
                parsed=parsed,
                output_root=args.output_root.resolve(),
                engine=engine,
                max_chars=args.max_chars,
                atom_max_tokens=args.atom_max_tokens,
                mode_max_tokens=args.mode_max_tokens,
                force=args.force,
            )
        )

    json_write(args.output_root.resolve() / "adapter_validation_summary.json", rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
