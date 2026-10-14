# OTSE V2 development evaluation

Place `evaluate_v2.py` in your OTSE repository root. This runner uses only `dev` rows from `data/corpus_v2/manifests/v2_split_metadata_balanced_proposed.csv` (450/60/60/46), verifies PDF hashes, and writes outputs separately. It never accesses test/reserve documents. It does not freeze the split or calculate factual F1 without reviewed gold.

## First five dev documents

```cmd
python evaluate_v2.py run --split dev --limit 5 --backend student --adapter-dir "training_outputs\qwen3-4b-instruct-2507-v2-pseudo-candidate\checkpoint-44"
python evaluate_v2.py run --split dev --limit 5 --backend teacher
python evaluate_v2.py report
```

You can use `--backend both` for one invocation; separate commands make it easier to debug and free GPU memory between backends. To expand to 60, use `--limit 60` with each command; completed document outputs are skipped automatically. If a model has context overflow, lower `--max-chars` (e.g. 12000) and rerun with `--force` after archiving the previous outputs. Do not raise context above 8448 for this controlled experiment.

Outputs: `data/corpus_v2/evaluation_dev/student/*.json`, `teacher/*.json`, `v2_eval_report.csv`, `v2_eval_summary.json`. `schema_valid` measures structural validity only. Inference timing is indicative, not an optimized or hardware-matched throughput benchmark. To measure accuracy, prepare independent reviewed gold for the same dev IDs. Do not use test data for model selection.
