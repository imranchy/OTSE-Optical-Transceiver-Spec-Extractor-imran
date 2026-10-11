from __future__ import annotations
from pathlib import Path
import sys
import fitz

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "gold_dataset" / "source_pdfs"
HOLD = ROOT / "gold_dataset" / "heldout_pdfs"


def inspect(pdf: Path):
    doc = fitz.open(pdf)
    chars = 0
    pages = len(doc)
    per_page = []
    for page in doc:
        t = page.get_text("text") or ""
        per_page.append(len(t.strip()))
        chars += len(t.strip())
    avg = chars / max(pages, 1)
    if chars == 0:
        status = "FAIL_NO_TEXT"
    elif avg < 250:
        status = "WARN_LOW_TEXT"
    else:
        status = "PASS"
    return pages, chars, avg, status


def run(folder: Path, label: str):
    print(f"\n=== {label} ===")
    pdfs = sorted(folder.glob("*.pdf"))
    failed = 0
    for p in pdfs:
        try:
            pages, chars, avg, status = inspect(p)
            print(f"{status:14} pages={pages:2d} chars={chars:7d} avg/page={avg:7.1f}  {p.name}")
            if status.startswith("FAIL"):
                failed += 1
        except Exception as e:
            failed += 1
            print(f"ERROR          {p.name}: {e}")
    return len(pdfs), failed

n_dev, bad_dev = run(DEV, "DEVELOPMENT")
n_hold, bad_hold = run(HOLD, "HELD OUT")
print(f"\nDevelopment: {n_dev} PDFs; hard failures: {bad_dev}")
print(f"Held out:    {n_hold} PDFs; hard failures: {bad_hold}")
if n_dev != 15:
    print(f"WARNING: expected 15 development PDFs, found {n_dev}")
if n_hold != 2:
    print(f"WARNING: expected 2 held-out PDFs, found {n_hold}")
if bad_dev or bad_hold:
    sys.exit(2)
