OTSE 4B/14B batch benchmark harness

Expected repo layout:
  gold_pipeline/benchmark_models.py
  gold_dataset/source_pdfs/       <- core 15 PDFs
  gold_dataset/heldout_pdfs/      <- long stress-test PDFs
  gold_dataset/benchmark_runs/    <- generated
  gold_dataset/reports/           <- generated

Create Ollama aliases:
  ollama create qwen3:4b-otse8k -f Modelfile.qwen3-4b-otse8k
  ollama create qwen3:14b-otse8k -f Modelfile.qwen3-14b-otse8k

Smoke test one PDF with both models:
  python gold_pipeline\benchmark_models.py --limit 1

Then run all core PDFs:
  python gold_pipeline\benchmark_models.py

The script is resumable: completed document/model pairs are skipped unless --force is used.
