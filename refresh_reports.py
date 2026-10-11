"""Rebuild reports/ so it matches runs/ exactly.

Reports are derived artifacts, so any run whose JSON changed needs its HTML
rebuilt, and any report whose run is gone is stale and misleading. Both
generators read whatever files are present and skip the rest, so a run without
modes.json still produces a usable pipeline report.

Usage:
    python refresh_reports.py --dry-run
    python refresh_reports.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from generate_modes_report import generate_modes_html
from generate_report import generate_html

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
REPORTS = HERE / "reports"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tag = "  [DRY RUN]" if args.dry_run else ""

    run_ids = sorted(p.name for p in RUNS.iterdir() if p.is_dir())
    REPORTS.mkdir(exist_ok=True)

    print("=" * 74)
    print("STALE REPORTS (run no longer present)" + tag)
    print("=" * 74)
    stale = 0
    for report in sorted(REPORTS.glob("*.html")):
        run_id = report.stem.replace("_modes", "").replace("_pipeline", "")
        if run_id not in run_ids:
            print(f"  remove {report.name}")
            stale += 1
            if not args.dry_run:
                report.unlink()
    if not stale:
        print("  none")

    print()
    print("=" * 74)
    print("REBUILD" + tag)
    print("=" * 74)
    built = 0
    for run_id in run_ids:
        run_dir = RUNS / run_id
        has_modes = (run_dir / "modes.json").exists()
        note = "" if has_modes else "   (no modes.json — pipeline report only)"
        print(f"  {run_id}{note}")
        if args.dry_run:
            continue
        (REPORTS / f"{run_id}_pipeline.html").write_text(
            generate_html(run_dir), encoding="utf-8")
        (REPORTS / f"{run_id}_modes.html").write_text(
            generate_modes_html(run_dir), encoding="utf-8")
        built += 1

    print()
    print(f"removed {stale} stale, rebuilt {built} runs x 2 reports")


if __name__ == "__main__":
    main()
