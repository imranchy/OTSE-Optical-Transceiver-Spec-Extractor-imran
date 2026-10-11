# Qwen 14B extraction workflow

This version uses Ollama at `http://127.0.0.1:11434` and selects `qwen3:14b`
when it is installed. The model is expected to run near the practical limit of
a 12 GB GPU. Keep the extraction context focused and use the default 4096
output-token limit.

## Start the application

```bat
ollama pull qwen3:14b
ollama serve
```

In a second terminal, from the repository root:

```bat
.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

## What changed

- Module-level extraction now uses a Pydantic-derived JSON Schema and rejects
  unknown output keys.
- The schema sent to the model removes examples and prose descriptions, because
  models can copy examples as false extracted values.
- Failed validation triggers one schema-repair request.
- Application-mode rows are extracted directly from their source table into a
  dedicated schema; they are no longer reconstructed from detached flat lists.
- Generation is deterministic by default: temperature 0, fixed seed, and Qwen
  reasoning disabled.

## Acceptance criteria

Do not judge success by whether a response merely looks plausible. For each
gold-labelled datasheet, measure:

1. JSON/schema validity rate.
2. Field-level precision and recall for module-level values.
3. Exact application-mode row count and row-level field accuracy.
4. Hallucination rate: values not explicitly supported by source text.

Keep PDFs, generated reports, and run folders out of commits unless you have
permission to redistribute them.
