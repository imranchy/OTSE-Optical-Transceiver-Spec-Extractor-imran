# OTSE: completed full-60 source adjudication and factual scoring

Extract this ZIP directly into the OTSE project root. Everything stays inside the new `otse_final_reference_v1` folder except this README and `RUN_OTSE_FULL60_FINAL.cmd`. Existing models, adapters, evaluator, predictions and old reference folders are not overwritten.

## Run once

From your existing activated Windows environment, in the project root:

```cmd
RUN_OTSE_FULL60_FINAL.cmd
```

Or run the scorer directly:

```cmd
python otse_final_reference_v1\score_full60.py --out paper_results\full60_ai_adjudicated_v1
```

No new packages, API key, model server or GPU are needed. It uses Python's standard library. It validates all 60 PDF hashes and all 180 saved prediction identities, then computes factual metrics and 10,000 paired document-bootstrap samples.

**Do not rerun the old `score_existing.py` or point an old scorer at `gold_review_verified_batch2.csv`: those are the old incomplete datasets.** This package uses the new complete JSONL reference, not the previous 63/80-row files.

## What is complete

All 60 PDFs / 282 pages have been source-reviewed. Every one of the 2,880 field records has a decision:

| Decision | Fields |
|---|---:|
| Supported source values | 1,375 |
| Confirmed absent under the documented field policy | 1,424 |
| Explicit source conflicts / schema ambiguities, excluded equally | 81 |
| Pending review | 0 |
| Scored across all 60 documents | 2,799 |

The reference is AI-assisted source adjudication, NOT independently human-certified gold. `PROTOCOL.md` explains exactly how to describe it. The 81 exceptions are completed review decisions, not unexamined rows. A source with conflicting values or several temperature grades cannot be made into a unique scalar by marking it verified.

## Files to inspect

- `otse_final_reference_v1\SOURCE_REVIEW.html`: open in a browser to navigate all 60 PDFs, all field decisions and page-linked evidence. No server needed.
- `gold_reference_full60.jsonl`: authoritative machine-readable reference; includes every field's decision, evidence, conditions and source hash.
- `gold_reference_full60.csv`: convenient readable export.
- `adjudication_exceptions.csv`: all 81 exclusions and reasons.
- `previous_annotation_changes.csv`: changes made when rechecking the earlier 80 annotated fields.
- `sources\`: all original test PDFs.
- `predictions\teacher`, `student`, `gguf`: frozen test predictions, no reruns.
- `results\`: precomputed full results, included so you can inspect them immediately.
- `PROTOCOL.md`: source rules, metrics, aliases, confidence intervals, limitations.

## New results folder

Running the command produces `paper_results\full60_ai_adjudicated_v1`:

- `factual_metrics.csv` and `.json`: precision, recall, micro/macro-F1, exact match, supported-field exact match, presence metrics and unmatched/omitted-value rates.
- `per_field_metrics.csv`, `per_group_metrics.csv`, `per_document_metrics.csv`.
- `field_value_scores.csv`: every scored model/field, matched values, unsupported/mismatched values and omissions.
- `teacher_error_summary.csv`, `teacher_error_fields.csv`: avoidance, regression and shared-failure counts, with explicit denominators and supported-field-only analysis.
- `confidence_intervals.json`: 95% paired document-bootstrap intervals and differences versus the teacher.
- `timing_and_schema.json`: original timings/schema counts plus disclosed contention sensitivity.
- `completion_summary.json`, `RESULTS_README.md`.

The complete results are lower than the former 63-field sample because the reference now covers every document and source-supported field. Do not reuse the earlier 88% F1 as the final test score.

## Integrity and tests

The command file runs the 25 included tests first. These cover BER precision, missing-key handling, alias boundaries, one-to-one matching, invalid numeric values, and the shipped full reference/prediction identities.

Source/model outputs are not changed. Hash validation detects input edits. An intended new annotation revision requires updating the reference and its manifest deliberately; do not bypass a mismatch silently.

The included manufacturer PDFs are for your local review. Do not automatically upload them publicly during GitHub cleanup. This package does not grant redistribution permissions or perform Git/model-hosting actions.
