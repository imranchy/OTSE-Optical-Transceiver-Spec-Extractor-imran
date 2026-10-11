# OTSE conservative evidence-validator diagnostic

Copy `validate_student_evidence.py` and `RUN_EVIDENCE_VALIDATOR.cmd` into the OTSE repository root. Run `RUN_EVIDENCE_VALIDATOR.cmd` or `python -u validate_student_evidence.py` from an activated environment.

This script reads the existing 45-task development JSONL and original student GGUF predictions. It never edits those inputs. It produces `training_outputs/otse-evidence-validator-dev/report.json`, `audit.csv`, and separate validated predictions. It uses OTSE's existing `scoring.score_one` and `scoring.metrics`.

Only three conservative automatic abstention rules are enabled: pure form-factor host-interface names, all-zero host-interface identifiers, and frequency predictions where the supplied evidence has no explicit GHz/THz unit. Temperature, wavelength and standards cases are flagged for review, not automatically changed. These rules are intentionally narrow, not guaranteed correct. **No silver-reference value is consulted when applying a rule.** The silver references are used afterward solely to calculate exploratory metrics.

Important: the development examples informed the choice of rules. These results cannot be called independent held-out accuracy. Validate changes against PDF source context and evaluate the frozen rules on a new, document-disjoint holdout before making a paper claim. Schema compliance and raw-model metrics must be reported separately from system-with-validator metrics.
