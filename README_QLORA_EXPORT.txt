OTSE GOLD -> QLoRA EXPORT (pilot v1)
===================================

Place these files in the repository root so gold_pipeline/*.py lands beside the
existing gold_pipeline scripts.

Prerequisites already expected locally:
  gold_dataset/benchmark_runs/
  gold_dataset/gold_candidates/accepted_fields.csv
  gold_dataset/gold_candidates/semantic_resolutions_applied.csv
  gold_dataset/gold_candidates/final_scalar_resolutions.csv
  gold_dataset/gold_candidates/final_mode_resolutions.csv
  gold_dataset/heldout_pdfs/

Run, in order:

  python gold_pipeline\merge_gold.py
  python gold_pipeline\build_splits.py
  python gold_pipeline\export_qlora.py

Expected split:
  12 train documents
   3 validation documents
   2 untouched held-out Cisco PDFs

Expected examples (approximately):
  Train:      36 atomic + reviewed mode-document examples (~41 total)
  Validation:  9 atomic + 1 reviewed coherent mode-document example (~10 total)

Do not change split_manifest.json after seeing QLoRA validation/test results.
Do not run the held-out PDFs until the adapted model and training choices are frozen.

Outputs:
  gold_dataset/final_gold/*.json
  gold_dataset/final_gold/final_gold_manifest.csv
  gold_dataset/splits/split_manifest.json
  gold_dataset/qlora/train.jsonl
  gold_dataset/qlora/validation.jsonl
  gold_dataset/qlora/train_atomic.jsonl
  gold_dataset/qlora/train_modes.jsonl
  gold_dataset/qlora/validation_atomic.jsonl
  gold_dataset/qlora/validation_modes.jsonl
  gold_dataset/qlora/dataset_summary.json

The exporter intentionally excludes notes/suggestions/provenance from atomic SFT
because those fields were not part of the verified gold-label target.
