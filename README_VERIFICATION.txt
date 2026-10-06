OTSE deterministic verification harness
======================================

Copy the files in gold_pipeline/ into your repository's existing gold_pipeline/ directory.

Run from repository root:

  python gold_pipeline\compare_models.py
  python gold_pipeline\evidence_validator.py
  python gold_pipeline\build_review_queue.py

Generated files under gold_dataset\reports\:
  field_comparison.csv
  comparison_summary.json
  verification_candidates.csv
  verification_summary.json
  auto_accept.csv
  review_queue.csv
  source_candidates.json
  mode_review_queue.csv
  review_queue.md

Policy:
- AUTO_ACCEPT requires 4B/14B agreement, successful atomic validation for both, and strong explicit source evidence.
- One-model-only values are never automatically gold, even when source-supported.
- Disagreements are never automatically gold.
- All modes require semantic row-level review.
- Obsolete smoke-test runs with <500 chars of source markdown are ignored automatically.

After running, ZIP gold_dataset\reports and upload it to ChatGPT for semantic adjudication of the remaining REVIEW cases.
