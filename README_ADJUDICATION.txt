OTSE Gold Candidate Adjudication
================================

Copy gold_pipeline/adjudicate_candidates.py into the repository root's gold_pipeline/ folder.

Prerequisites:
  python gold_pipeline/compare_models.py
  python gold_pipeline/evidence_validator.py
  python gold_pipeline/build_review_queue.py

Run:
  python gold_pipeline/adjudicate_candidates.py

Outputs:
  gold_dataset/gold_candidates/accepted_fields.csv
  gold_dataset/gold_candidates/unresolved.csv
  gold_dataset/gold_candidates/ignored_metadata.csv
  gold_dataset/gold_candidates/adjudication_summary.json
  gold_dataset/gold_candidates/partial_gold_specs/*.json

Policy:
- Existing strong 4B/14B agreement is accepted.
- Connector/fiber/coherent-modulation normalization can turn harmless wording conflicts into agreement.
- Strong 14B-only evidence is promoted only for low-ambiguity scalar/identity fields.
- 4B-only promotion is stricter.
- Title-backed form-factor conflicts can be resolved.
- Explicit operating-temperature context and transmitter-power context can resolve a small number of conflicts.
- Reach, conditional optical limits, host/media relationships, and modes are not guessed.
- notes/suggestions/provenance metadata are excluded from QLoRA target candidates.
- partial_gold_specs are NOT final training gold until unresolved.csv is adjudicated.
