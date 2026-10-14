from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gold_pipeline.benchmark_models import (
    check_ollama,
    collect_existing_summary,
    json_write,
    model_slug,
    run_one_model,
    utc_now,
    write_summary_csv,
)
from pipeline.atom_extractor import BASE_URL
from pipeline.pdf_parser import ParsedDatasheet, ParsedPage, ParsedTable

LOGGER = logging.getLogger("otse.stock_instruct_validation")

DEFAULT_MODEL = "qwen3:4b-instruct-otse8448"
CONTEXT_LIMIT = 8448
ATOM_MAX_TOKENS = 3072
MODE_MAX_TOKENS = 2048
MAX_CHARS = 120_000


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild_parsed(doc_dir: Path) -> ParsedDatasheet:
    source_dir = doc_dir / "source"
    content_path = source_dir / "content.md"
    summary_path = source_dir / "parsed_summary.json"
    tables_path = source_dir / "tables.json"

    missing = [p for p in (content_path, summary_path, tables_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen source artifacts for "
            f"{doc_dir.name}: {', '.join(str(p) for p in missing)}"
        )

    markdown = content_path.read_text(encoding="utf-8")
    summary = load_json(summary_path)
    table_rows = load_json(tables_path)

    pages = [ParsedPage(index=i, text="") for i in range(int(summary.get("page_count", 0)))]
    tables = [
        ParsedTable(
            page_index=int(t.get("page_index", 0)),
            bbox=tuple(t.get("bbox", (0, 0, 0, 0))),
            rows=t.get("rows", []),
            label=t.get("label"),
        )
        for t in table_rows
    ]

    # source_path is metadata only here. We intentionally do NOT reparse PDFs.
    return ParsedDatasheet(
        source_path=Path(summary.get("source_file", f"{doc_dir.name}.pdf")),
        pages=pages,
        tables=tables,
        markdown=markdown,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run stock Qwen3-4B-Instruct-2507 on ONLY the frozen 3-document validation split."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    split_path = REPO_ROOT / "gold_dataset" / "splits" / "split_manifest.json"
    benchmark_dir = REPO_ROOT / "gold_dataset" / "benchmark_runs"
    report_dir = REPO_ROOT / "gold_dataset" / "reports_instruct_validation"

    manifest = load_json(split_path)
    validation_ids = list(manifest["validation"])
    heldout = list(manifest.get("heldout_pdfs", []))

    if len(validation_ids) != 3:
        raise RuntimeError(f"Expected exactly 3 frozen validation documents, found {len(validation_ids)}")

    LOGGER.info("Frozen validation documents (%d): %s", len(validation_ids), ", ".join(validation_ids))
    LOGGER.info("Held-out PDFs are NOT accessed by this script: %s", heldout)
    LOGGER.info(
        "Frozen inference settings: context=%d atom_max_tokens=%d mode_max_tokens=%d temperature=0",
        CONTEXT_LIMIT,
        ATOM_MAX_TOKENS,
        MODE_MAX_TOKENS,
    )

    environment = check_ollama(args.base_url, [args.model])
    environment.update(
        {
            "benchmark_started_at_utc": utc_now(),
            "model": args.model,
            "split": "validation",
            "validation_document_ids": validation_ids,
            "heldout_pdfs_not_accessed": heldout,
            "context_limit": CONTEXT_LIMIT,
            "atom_max_tokens": ATOM_MAX_TOKENS,
            "mode_max_tokens": MODE_MAX_TOKENS,
            "input_source": "existing benchmark_runs/<document>/source/content.md",
            "pdf_reparse": False,
        }
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    json_write(report_dir / "stock_instruct_environment.json", environment)

    results = []
    for index, doc_id in enumerate(validation_ids, start=1):
        doc_dir = benchmark_dir / doc_id
        if not doc_dir.exists():
            raise FileNotFoundError(f"Validation benchmark directory missing: {doc_dir}")

        LOGGER.info("[%d/%d] %s", index, len(validation_ids), doc_id)
        parsed = rebuild_parsed(doc_dir)
        result = run_one_model(
            parsed=parsed,
            doc_dir=doc_dir,
            model=args.model,
            base_url=args.base_url,
            max_chars=MAX_CHARS,
            atom_max_tokens=ATOM_MAX_TOKENS,
            mode_max_tokens=MODE_MAX_TOKENS,
            context_limit=CONTEXT_LIMIT,
            force=args.force,
        )
        result["document_id"] = doc_id
        results.append(result)

        # Dedicated validation report; do not overwrite the historical full-corpus report.
        write_summary_csv(report_dir / "stock_instruct_validation_summary.csv", results)
        json_write(report_dir / "stock_instruct_validation_summary.json", results)

    statuses = {row.get("status") for row in results}
    LOGGER.info("Stock Instruct validation complete. Statuses: %s", sorted(statuses))
    LOGGER.info("Results stored under each validation document in: %s", benchmark_dir)
    LOGGER.info("Model subdirectory: %s", model_slug(args.model))
    LOGGER.info("Summary: %s", report_dir / "stock_instruct_validation_summary.csv")


if __name__ == "__main__":
    main()
