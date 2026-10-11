from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# Allow execution as: python gold_pipeline\benchmark_models.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.atom_extractor import BASE_URL, extract_atoms
from pipeline.mode_linker import store_modes, synthesize_modes
from pipeline.pdf_parser import ParsedDatasheet, parse_pdf

LOGGER = logging.getLogger("otse.gold.benchmark")

DEFAULT_MODELS = ("qwen3:4b-otse8k", "qwen3:14b-otse8k")
EMPTY_VALUES = (None, "", [], {})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(text: str, max_len: int = 70) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return (text[:max_len].rstrip("-") or "datasheet")


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def populated_top_level_fields(data: dict[str, Any] | None) -> int:
    if not data:
        return 0
    return sum(1 for value in data.values() if value not in EMPTY_VALUES)


def approximate_tokens(text: str) -> int:
    # Conservative language-independent estimate for flagging context risk only.
    # This is NOT used as an accuracy metric.
    return max(1, (len(text) + 3) // 4)


def ollama_root_from_base_url(base_url: str) -> str:
    return base_url[:-3] if base_url.endswith("/v1") else base_url


def check_ollama(base_url: str, requested_models: list[str]) -> dict[str, Any]:
    root = ollama_root_from_base_url(base_url)
    version_resp = requests.get(f"{root}/api/version", timeout=10)
    version_resp.raise_for_status()
    models_resp = requests.get(f"{root}/v1/models", timeout=20)
    models_resp.raise_for_status()
    installed = sorted(item["id"] for item in models_resp.json().get("data", []))
    missing = [model for model in requested_models if model not in installed]
    if missing:
        raise RuntimeError(
            "Requested Ollama models are not installed: "
            + ", ".join(missing)
            + "\nInstalled models: "
            + ", ".join(installed)
        )
    return {
        "ollama_version": version_resp.json().get("version"),
        "installed_models": installed,
        "base_url": base_url,
    }


def ollama_model_info(base_url: str, model: str) -> dict[str, Any]:
    root = ollama_root_from_base_url(base_url)
    try:
        response = requests.post(f"{root}/api/show", json={"model": model}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return {
            "details": payload.get("details", {}),
            "parameters": payload.get("parameters", ""),
            "model_info": payload.get("model_info", {}),
        }
    except Exception as exc:  # diagnostic only
        return {"diagnostic_error": str(exc)}


def ollama_process_info(base_url: str, model: str) -> dict[str, Any]:
    root = ollama_root_from_base_url(base_url)
    try:
        response = requests.get(f"{root}/api/ps", timeout=10)
        response.raise_for_status()
        items = response.json().get("models", [])
        for item in items:
            if item.get("name") == model or item.get("model") == model:
                return item
        return {}
    except Exception as exc:  # diagnostic only
        return {"diagnostic_error": str(exc)}


def parsed_summary(parsed: ParsedDatasheet, sha256: str) -> dict[str, Any]:
    markdown = parsed.markdown or parsed.full_text
    return {
        "source_file": parsed.source_path.name,
        "source_sha256": sha256,
        "page_count": parsed.page_count,
        "table_count": len(parsed.tables),
        "markdown_chars": len(markdown),
        "approx_input_tokens": approximate_tokens(markdown),
        "parsed_at_utc": utc_now(),
    }


def save_parsed_artifacts(doc_dir: Path, parsed: ParsedDatasheet, summary: dict[str, Any]) -> None:
    source_dir = doc_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    markdown = parsed.markdown or parsed.full_text
    (source_dir / "content.md").write_text(markdown, encoding="utf-8")
    json_write(source_dir / "parsed_summary.json", summary)
    json_write(
        source_dir / "tables.json",
        [
            {
                "page_index": table.page_index,
                "bbox": list(table.bbox),
                "label": table.label,
                "rows": table.rows,
            }
            for table in parsed.tables
        ],
    )


def build_document_id(pdf: Path, sha256: str) -> str:
    return f"{slugify(pdf.stem)}-{sha256[:8]}"


def run_one_model(
    *,
    parsed: ParsedDatasheet,
    doc_dir: Path,
    model: str,
    base_url: str,
    max_chars: int,
    atom_max_tokens: int,
    mode_max_tokens: int,
    context_limit: int,
    force: bool,
) -> dict[str, Any]:
    out_dir = doc_dir / model_slug(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists() and not force:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"success", "partial"}:
            LOGGER.info("SKIP completed %s / %s", parsed.source_path.name, model)
            return existing

    markdown = parsed.markdown or parsed.full_text
    metrics: dict[str, Any] = {
        "status": "running",
        "started_at_utc": utc_now(),
        "source_file": parsed.source_path.name,
        "model": model,
        "base_url": base_url,
        "temperature": 0.0,
        "atom_max_tokens": atom_max_tokens,
        "mode_max_tokens": mode_max_tokens,
        "max_chars": max_chars,
        "page_count": parsed.page_count,
        "table_count": len(parsed.tables),
        "markdown_chars": len(markdown),
        "approx_input_tokens": approximate_tokens(markdown),
        "configured_context_limit": context_limit,
        "context_risk": approximate_tokens(markdown) > int(context_limit * 0.80),
        "model_info": ollama_model_info(base_url, model),
    }
    json_write(metrics_path, metrics)

    total_started = time.perf_counter()
    try:
        LOGGER.info("ATOMS %s / %s", parsed.source_path.name, model)
        atom_started = time.perf_counter()
        specs, raw_atoms = extract_atoms(
            parsed,
            model_id=model,
            max_chars=max_chars,
            base_url=base_url,
            temperature=0.0,
            max_tokens=atom_max_tokens,
        )
        atom_seconds = time.perf_counter() - atom_started
        json_write(out_dir / "raw_llm.json", raw_atoms)

        specs_dict: dict[str, Any] | None = None
        if specs is not None:
            specs_dict = specs.model_dump(mode="json")
            json_write(out_dir / "specs.json", specs_dict)

        metrics.update(
            {
                "atomic_seconds": round(atom_seconds, 4),
                "atomic_validation_passed": specs is not None,
                "populated_spec_fields": populated_top_level_fields(specs_dict or raw_atoms),
                "raw_atom_fields": populated_top_level_fields(raw_atoms),
            }
        )

        mode_seconds: float | None = None
        modes_count = 0
        modes_validation_passed: bool | None = None

        if specs_dict is not None:
            LOGGER.info("MODES %s / %s", parsed.source_path.name, model)
            mode_started = time.perf_counter()
            modes, multi, general = synthesize_modes(
                specs_dict,
                parsed,
                run_dir=out_dir,
                model_id=model,
                base_url=base_url,
                temperature=0.0,
                max_tokens=mode_max_tokens,
            )
            mode_seconds = time.perf_counter() - mode_started
            modes_validation_passed = modes is not None
            if modes is not None:
                store_modes(out_dir, modes, multi, general)
                modes_count = len(modes.get("modes", []))
            else:
                json_write(
                    out_dir / "mode_atoms.json",
                    {"multi_valued": multi, "general": general},
                )
        else:
            LOGGER.warning("Skipping mode extraction because atomic validation failed: %s", parsed.source_path.name)

        metrics.update(
            {
                "mode_seconds": None if mode_seconds is None else round(mode_seconds, 4),
                "modes_validation_passed": modes_validation_passed,
                "modes_count": modes_count,
                "total_seconds": round(time.perf_counter() - total_started, 4),
                "ollama_process_after": ollama_process_info(base_url, model),
                "finished_at_utc": utc_now(),
            }
        )
        metrics["status"] = "success" if specs is not None else "partial"
        json_write(metrics_path, metrics)
        return metrics

    except Exception as exc:
        metrics.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "total_seconds": round(time.perf_counter() - total_started, 4),
                "finished_at_utc": utc_now(),
            }
        )
        (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        json_write(metrics_path, metrics)
        LOGGER.exception("FAILED %s / %s", parsed.source_path.name, model)
        return metrics


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_file",
        "document_id",
        "model",
        "status",
        "page_count",
        "table_count",
        "markdown_chars",
        "approx_input_tokens",
        "configured_context_limit",
        "context_risk",
        "atomic_validation_passed",
        "populated_spec_fields",
        "modes_validation_passed",
        "modes_count",
        "atomic_seconds",
        "mode_seconds",
        "total_seconds",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_existing_summary(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_dir.glob("*/*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["document_id"] = metrics_path.parents[1].name
            rows.append(metrics)
        except Exception:
            LOGGER.warning("Could not read metrics: %s", metrics_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch OTSE benchmark: parse each PDF once, then run identical extraction with multiple Ollama models."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "gold_dataset" / "source_pdfs",
        help="Directory containing the core PDF corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "gold_dataset" / "benchmark_runs",
        help="Directory for per-document benchmark artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPO_ROOT / "gold_dataset" / "reports",
        help="Directory for summary CSV/JSON files.",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--max-chars", type=int, default=120_000)
    parser.add_argument("--atom-max-tokens", type=int, default=8192)
    parser.add_argument("--mode-max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N PDFs (smoke testing).")
    parser.add_argument("--force", action="store_true", help="Rerun completed model/document pairs.")
    parser.add_argument("--skip-modes", action="store_true", help="Reserved for debugging; currently not used.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.skip_modes:
        raise SystemExit("--skip-modes is reserved; remove it for the production benchmark.")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(source_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {source_dir}")

    environment = check_ollama(args.base_url, args.models)
    environment.update(
        {
            "benchmark_started_at_utc": utc_now(),
            "models": args.models,
            "source_dir": str(source_dir),
            "pdf_count": len(pdfs),
            "context_limit": args.context_limit,
        }
    )
    json_write(report_dir / "benchmark_environment.json", environment)

    # Parse ONCE. The same ParsedDatasheet object/Markdown is reused for all models.
    parsed_docs: list[tuple[Path, str, ParsedDatasheet, Path]] = []
    LOGGER.info("Parsing %d PDFs once before model inference", len(pdfs))
    for index, pdf in enumerate(pdfs, start=1):
        parse_started = time.perf_counter()
        sha = sha256_file(pdf)
        parsed = parse_pdf(pdf)
        doc_id = build_document_id(pdf, sha)
        doc_dir = output_dir / doc_id
        summary = parsed_summary(parsed, sha)
        summary.update(
            {
                "document_id": doc_id,
                "parse_seconds": round(time.perf_counter() - parse_started, 4),
                "corpus_index": index,
            }
        )
        save_parsed_artifacts(doc_dir, parsed, summary)
        parsed_docs.append((pdf, doc_id, parsed, doc_dir))

    # Model is the outer loop to avoid repeatedly swapping 4B <-> 14B in VRAM.
    for model in args.models:
        LOGGER.info("========== MODEL %s ==========", model)
        for ordinal, (pdf, doc_id, parsed, doc_dir) in enumerate(parsed_docs, start=1):
            LOGGER.info("[%d/%d] %s", ordinal, len(parsed_docs), pdf.name)
            metrics = run_one_model(
                parsed=parsed,
                doc_dir=doc_dir,
                model=model,
                base_url=args.base_url,
                max_chars=args.max_chars,
                atom_max_tokens=args.atom_max_tokens,
                mode_max_tokens=args.mode_max_tokens,
                context_limit=args.context_limit,
                force=args.force,
            )
            metrics["document_id"] = doc_id
            rows = collect_existing_summary(output_dir)
            write_summary_csv(report_dir / "benchmark_summary.csv", rows)

    rows = collect_existing_summary(output_dir)
    write_summary_csv(report_dir / "benchmark_summary.csv", rows)
    json_write(report_dir / "benchmark_summary.json", rows)
    LOGGER.info("Benchmark complete. Summary: %s", report_dir / "benchmark_summary.csv")


if __name__ == "__main__":
    main()
