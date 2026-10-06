OTSE experiment corpus builder

This package already contains:
- 5 development PDFs that passed native-text checks
- 2 held-out Cisco PDFs

Run this from the repository root in PowerShell:
  powershell -ExecutionPolicy Bypass -File .\download_official_native_pdfs.ps1

Then preflight all PDFs:
  python gold_pipeline\preflight_corpus.py

Expected final corpus:
- gold_dataset\source_pdfs: 15 PDFs
- gold_dataset\heldout_pdfs: 2 PDFs

Only after preflight passes should you run the model benchmark.
