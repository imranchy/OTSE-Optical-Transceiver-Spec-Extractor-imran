---
Base Model: Qwen/Qwen3-4B-Instruct-2507
tags:
  - gguf
  - qlora
  - optical-transceivers
  - information-extraction
  - structured-extraction
  - telecommunications
pipeline_tag: text-generation
---

# OTSE-Qwen3-4B-GGUF

**Fine-tuned small language model for structured optical transceiver specification extraction.**

This model is part of the Optical Transceiver Specification Extractor (OTSE) research project.

It is based on Qwen3-4B-Instruct-2507, adapted using QLoRA with teacher-generated optical-domain pseudolabels, and distributed in Q4_K_M GGUF format for local deployment.

## Model Information

| Property | Value |
|---|---|
| Base model | Qwen3-4B-Instruct-2507 |
| Parameters | 4.0B |
| Training method | QLoRA |
| Training examples | 745 |
| Unique training document IDs | 391 |
| LoRA rank | 32 |
| Quantization | Q4_K_M |
| Extraction schema | 48 atomic fields |
| Intended runtime | Ollama / llama.cpp |

## Evaluation

Matched development evaluation on 45 field-group extraction tasks covering 15 datasheets:

| Metric | Qwen3-14B Teacher | OTSE Qwen3-4B Student |
|---|---:|---:|
| Precision | 96.72% | 76.67% |
| Recall | 93.65% | 85.19% |
| Micro-F1 | 95.16% | 80.70% |
| Mean inference-request latency | 10.17 s | 3.25 s |

The evaluation uses verified references. 

## Local Deployment

Download `otse-qwen4b-Q4_K_M.gguf` and place it in the `deployment/` directory of the OTSE GitHub repository.

Run:

```bash
ollama create otse-qwen4b-final -f Modelfile.otse-final
```

Launch the application:

```bash
streamlit run app.py
```

The application supports PDF parsing, structured specification extraction, schema validation and optional human correction.

## Model Integrity

SHA256:

`5f5296c65b9dd961f35748873360403fc1caee534bb5a842e632f11c81c5d8b2`

## Limitations

The model may misinterpret technical units, confuse optical parameters, or associate specifications with incorrect operating modes. Valid JSON does not guarantee factual correctness.

Human review is recommended before using extracted specifications for engineering decisions.

## Community Contributions

We welcome source-evidenced optical transceiver annotations, vendor datasheet test cases, extraction corrections and reproducible evaluation contributions.

Please contribute through GitHub Issues or Pull Requests. Only submit material you have permission to redistribute.

## Source Code

[OTSE GitHub Research Release](https://github.com/imranchy/OTSE-Optical-Transceiver-Spec-Extractor-imran/tree/release/ofc2027-public)