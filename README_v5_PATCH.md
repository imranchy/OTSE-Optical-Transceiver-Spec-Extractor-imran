# OTSE Qwen-14B general extraction patch v5

This is a **drop-in patch**, not a replacement repository. Copy the included `pipeline/` files into the existing OTSE project and keep the rest of your current project unchanged.

## What changed

- Replaces one giant atom-extraction prompt with 3 focused passes.
- Keeps `extract_atoms(...) -> (specs, raw_dict)` unchanged, so `app.py` does not need modification.
- Adds conservative deterministic recovery for facts explicitly printed in the parsed text.
- Adds an evidence coverage gate so an all-null/empty object cannot be treated as a successful extraction when explicit facts are present.
- Adds one targeted retry only for evidence-backed missing fields.
- Uses `temperature=0.0` by default for repeatability.
- Does **not** add agents, web tools, vendor-specific knowledge, or hidden inference.

## Install

From the project root, back up your current files, then copy:

- `pipeline/atom_extractor.py`
- `pipeline/extraction_quality.py`

No change is required in `app.py` for this patch.

## Local checks

```bat
python -m py_compile pipeline\atom_extractor.py pipeline\extraction_quality.py
python -c "from pipeline.atom_extractor import extract_atoms; from pipeline.extraction_quality import deterministic_hints; print('imports OK')"
```

If pytest is available:

```bat
pytest -q tests\test_extraction_quality.py
```

Then start your existing llama-server with your Qwen model and run:

```bat
streamlit run app.py
```

Re-test the Smartoptics `SO-TQSFPDD4CCZRP` file first. An extraction that returns null/empty for explicit identity/spectrum facts should now be rejected or recovered rather than displayed as a valid success.

## Why no fine-tuning yet

Fine-tuning should be evaluated only after this deterministic + focused-prompt baseline is measured on a held-out, vendor-diverse PDF set. QLoRA should use reviewed input -> JSON examples, not raw PDFs alone.
