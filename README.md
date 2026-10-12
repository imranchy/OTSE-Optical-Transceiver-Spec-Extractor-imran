# OTSE: Optical Transceiver Specification Extractor

**Teacher-Guided QLoRA Adaptation of a Small Language Model for Structured Optical Transceiver Datasheet Extraction**

OTSE is an open-source research prototype for extracting structured technical specifications from optical transceiver datasheets using locally deployed language models.

The project combines PDF processing, domain-specific structured extraction, teacher-guided pseudolabeling, QLoRA fine-tuning, schema validation, and optional human review.

The primary deployment model is a fine-tuned **Qwen3-4B-Instruct-2507**, quantized to GGUF Q4_K_M. A larger Qwen3-14B model serves as the teacher and experimental reference.

## Download the Fine-Tuned Models

Both the original fine-tuned deployment model and its QLoRA adapter are available on Hugging Face.

| Model | Description | Download |
|---|---|---|
| OTSE-Qwen3-4B-GGUF | Q4_K_M quantized model for local inference | [Hugging Face GGUF](https://huggingface.co/diptoimran/OTSE-Qwen3-4B-GGUF) |
| OTSE-Qwen3-4B-QLoRA | Original LoRA adapter for research and further adaptation | [Hugging Face Adapter](https://huggingface.co/diptoimran/OTSE-Qwen3-4B-QLoRA) |

To deploy the model, download `otse-qwen4b-Q4_K_M.gguf` into the `deployment/` directory and register it using Ollama:

```bash
ollama create otse-qwen4b-final -f Modelfile.otse-final
```

Launch the application:

```bash
streamlit run app.py
```

The application provides structured extraction, schema validation, and optional human correction.

## 1. Overview

Optical transceiver datasheets contain technical information distributed across tables, paragraphs, product variants, and operating configurations. Extracting these parameters manually is time-consuming and can be inconsistent across vendors.

OTSE aims to automate this process while preserving the ability to inspect and correct extracted values.

The system supports:

- PDF parsing and table extraction.
- Structured optical specification extraction using a 48-field atomic schema.
- Optical-domain terminology including OpenZR+, OIF 400ZR, FEC, modulation formats, host interfaces, and wavelength information.
- Local inference with a fine-tuned Qwen3-4B model.
- Schema validation using Pydantic.
- Optional application-mode extraction.
- Human inspection, editing, and correction through a Streamlit interface.
- Export and storage of structured results.
- Research utilities for automated annotation, QLoRA training, evaluation, and GGUF conversion.

Extracted specifications should be verified against source documentation before use in operational or safety-critical workflows.

## 2. System Architecture

The application follows this general workflow:

```text
Optical Transceiver Datasheet (PDF)
                |
                v
        PDF Parsing
                |
                v
      Text and Table Extraction
                |
                v
      Atomic Field Extraction
                |
                v
    Fine-tuned Qwen3-4B Student
                |
                v
        Schema Validation
                |
                v
       Structured JSON Output
                |
                v
       Optional Human Review
                |
                v
       Corrected Specifications
                |
                v
       Storage and Reporting
```

Application-mode extraction is implemented as a separate optional stage. Its performance is not established by the atomic-field benchmark reported below.

### Main Components

| Component | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `pipeline/run.py` | Command-line PDF extraction |
| `pipeline/pdf_parser.py` | PDF and table parsing |
| `pipeline/atom_extractor.py` | Atomic specification extraction |
| `pipeline/mode_linker.py` | Optional application-mode extraction |
| `transceiver_models.py` | Specification schemas |
| `otse_auto/` | Automated annotation and research pipeline |
| `training/` | QLoRA training code and configurations |
| `validate_student_evidence.py` | Optional evidence-validation research utility |
| `results/ofc2027/` | Selected experimental results |
| `Modelfile.otse-final` | Local Ollama deployment configuration |

## 3. Installation

### Requirements

- Python 3.11 or a compatible version.
- Ollama installed and running.
- Sufficient RAM and storage for local GGUF inference.
- The OTSE fine-tuned GGUF model.
- A CUDA-capable GPU is recommended for accelerated inference.
- A suitable CUDA training environment is required for QLoRA fine-tuning.

The released GGUF can also be used with compatible llama.cpp runtimes.

### Clone the Repository

Clone the public research-release branch:

```bash
git clone --branch release/ofc2027-public --single-branch https://github.com/imranchy/OTSE-Optical-Transceiver-Spec-Extractor-imran.git
cd OTSE-Optical-Transceiver-Spec-Extractor-imran
```

### Create a Python Environment

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The runtime requirements include the `openai` Python client for communication with the local Ollama-compatible API.

The research training pipeline requires additional packages including PyTorch, Transformers, PEFT, Accelerate, and bitsandbytes. Their compatible versions depend on the CUDA environment.

## 4. Model Deployment

### Primary Student Model

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3-4B-Instruct-2507 |
| Fine-tuning | QLoRA |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Deployment format | GGUF |
| Quantization | Q4_K_M |
| Context setting | 8192 tokens |
| Temperature | 0 |
| Local Ollama name | `otse-qwen4b-final` |

### Model Integrity

The final original student GGUF is named:

`otse-qwen4b-Q4_K_M.gguf`

SHA256:

```text
5f5296c65b9dd961f35748873360403fc1caee534bb5a842e632f11c81c5d8b2
```

The checksum is also recorded in `models/original_student_sha256.txt`.

The GGUF is distributed separately from GitHub due to its size.

**Download the verified GGUF model:**

[OTSE-Qwen3-4B-GGUF](https://huggingface.co/diptoimran/OTSE-Qwen3-4B-GGUF)

**Download the original QLoRA adapter:**

[OTSE-Qwen3-4B-QLoRA](https://huggingface.co/diptoimran/OTSE-Qwen3-4B-QLoRA)

### Install the Model Locally

After downloading the GGUF, create the `deployment/` directory if it does not exist, and place the model at:

```text
deployment/otse-qwen4b-Q4_K_M.gguf
```

The repository's `Modelfile.otse-final` contains:

```text
FROM ./deployment/otse-qwen4b-Q4_K_M.gguf
PARAMETER num_ctx 8192
PARAMETER temperature 0
```

Register the model:

```bat
ollama create otse-qwen4b-final -f Modelfile.otse-final
```

Verify:

```bat
ollama show otse-qwen4b-final
```

The original model has been successfully imported into Ollama and verified with a JSON generation smoke test.

### Fine-Tuning Adapter

The original QLoRA adapter is available at:

[OTSE-Qwen3-4B-QLoRA](https://huggingface.co/diptoimran/OTSE-Qwen3-4B-QLoRA)

The adapter contains the LoRA weights trained using 745 teacher-generated extraction examples from 391 unique transceiver datasheets.

Researchers can use the adapter with the original `Qwen/Qwen3-4B-Instruct-2507` base model for further adaptation.

The adapter is not a standalone GGUF model; loading it requires the compatible base model and PEFT tooling.

## 5. Running OTSE

### Command-Line Extraction

Ensure Ollama is running and the model has been registered.

Run:

```bat
python -m pipeline.run "path\to\datasheet.pdf" --model otse-qwen4b-final
```

The pipeline parses the PDF, extracts atomic specifications, validates the output, and stores the run.

Generated results are saved under `runs/`, typically including:

- `content.md`: Parsed datasheet content.
- `parsed.json`: PDF parsing metadata.
- `raw_llm.json`: Extracted raw parameter values.
- `specs.json`: Validated structured specifications.
- `meta.json`: Model, source, validation, and timing metadata.

### Streamlit Application

Launch:

```bat
streamlit run app.py
```

The application provides a browser-based workflow for uploading datasheets, reviewing parsed text, extracting parameters, editing structured results, and generating reports.

The default model is `otse-qwen4b-final`, with the option to select another locally available Ollama model.

### Human-in-the-Loop Review

OTSE includes an optional manual verification workflow.

Users can inspect the extracted specifications, compare them with the source document, modify incorrect or incomplete JSON values, and save the corrected output.

The interface also permits editing parsed text before extraction.

**Important:** Schema validation verifies output structure and data types. It does not establish that every extracted value is factually correct. Human review is optional and is not automatically performed on benchmark predictions.

## 6. Dataset and Training Methodology

### Corpus Preparation

The research corpus manifest contains 616 PDF records with the following proposed partition:

| Partition | Documents |
|---|---:|
| Training | 450 |
| Development | 60 |
| Test | 60 |
| Reserve | 46 |
| **Total** | **616** |

The manifest records document identifiers, file hashes, family identifiers, technology, form factor, and split assignments.

The original student training dataset was generated using teacher-guided pseudolabeling and structured extraction tasks.

### Original QLoRA Dataset

| Property | Count |
|---|---:|
| Training examples | 745 |
| Unique training document IDs | 391 |
| Validation examples | 78 |
| Unique validation document IDs | 43 |

The examples are structured into atomic extraction tasks corresponding to groups of related optical parameters.

Training labels were generated through an automated teacher-assisted workflow. The original training dataset includes records marked as unreviewed 14B pseudolabels; these must not be described as independently human-verified gold annotations.

### QLoRA Configuration

The original student training configuration is preserved at:

`training/config_v2_expanded_745.json`

Key settings:

| Parameter | Value |
|---|---|
| Base model | Qwen3-4B-Instruct-2507 |
| Training epochs | 2 |
| Learning rate | 2e-5 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Maximum sequence length | 8448 |
| Gradient accumulation | 8 |
| Quantization for training | 4-bit NF4 |
| Random seed | 42 |

The final adapter was merged into the base model and converted to a quantized GGUF for deployment.

The `otse_auto/` package also contains a subsequent automated research workflow. Its default settings and candidate-model experiments should not be confused with the configuration used to train the original released student.

## 7. Experimental Evaluation

### Matched Teacher–Student Comparison

The Qwen3-14B teacher and original Qwen3-4B student were compared using matched development extraction tasks and consistent field-scoring rules.

The evaluation consisted of 45 atomic extraction tasks covering 15 distinct development datasheets.

| Metric | Qwen3-14B Teacher | Qwen3-4B Student |
|---|---:|---:|
| Precision | 96.72% | 76.67% |
| Recall | 93.65% | 85.19% |
| Micro-F1 | 95.16% | 80.70% |
| Schema-key failures | 0 | 0 |
| Mean inference-request time | 10.17 s | 3.25 s |

The student achieved approximately **3.13× lower mean inference-request latency** in this comparison.

These measurements represent the recorded request-level inference configuration, not complete end-to-end PDF processing times.

The teacher and student were evaluated against machine-generated silver references. Consequently, the reported scores should not be interpreted as independently verified real-world extraction accuracy.

### Exploratory Evidence Validation

An optional evidence-validation experiment improved the student development micro-F1 from approximately 80.70% to 83.42%.

This was an exploratory development-set result. It is not presented as an independently validated production improvement.

### End-to-End Demonstration

The final Qwen3-4B model was also tested on a four-page Smartoptics coherent transceiver datasheet using the existing PDF extraction pipeline.

Recorded timings:

| Stage | Time |
|---|---:|
| PDF parsing | 11.50 s |
| Atomic extraction | 16.95 s |
| Result storage | 0.014 s |
| Total | 28.47 s |

The output passed Pydantic schema validation and was saved successfully.

This single-document smoke test demonstrates pipeline integration, not a general accuracy or latency guarantee.

## 8. Reproducibility

The repository preserves selected components needed to inspect the original research:

- Original QLoRA training configuration.
- Teacher-guided annotation and evaluation code.
- Corpus split metadata.
- Matched teacher–student benchmark summaries.
- Model SHA256 fingerprint.
- Source-grounding and schema-validation utilities.
- Unit tests and self-tests.

Run the tests with:

```bat
python -m pytest -q
python -m otse_auto selftest
```

At the release checkpoint, the two suites passed 6 and 38 tests respectively.

The original training JSONL files and source PDFs are not included in the public release. Their availability is subject to redistribution permissions.

A complete independent reproduction requires access to the appropriate source documents, training records, model weights, compatible dependencies, and the frozen evaluation references.

## 9. Limitations

OTSE currently has several limitations:

- The smaller student does not match the teacher's measured development F1.
- Some numerical values may be assigned to incorrect fields.
- Source units and operating-mode associations may be misinterpreted.
- Valid JSON does not guarantee factual correctness.
- Evaluation relies on machine-generated silver references.
- Application-mode extraction has not been established by the atomic extraction benchmark.
- Model performance may vary across vendors, datasheet formats, and inference runtimes.
- The human-review workflow is available but has not been separately evaluated for correction accuracy or review time.

Users should inspect uncertain values against the original datasheet.

## 10. Future Work

Planned research directions include improving the quality of source-verified training annotations, expanding cross-vendor evaluation, investigating schema-aware domain adaptation, and improving operating-mode association.

Further work may also explore larger models, additional QLoRA training, and server-GPU-based experimentation.

Another important direction is measuring the effectiveness and time cost of human verification and correction.

Community contributions of permitted,