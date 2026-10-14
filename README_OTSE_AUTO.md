# OTSE automated source-grounded QLoRA workflow

**No manual field verification or APPROVED flags are required.** This package replaces the blocked human-review workflow with a separate, explicitly machine-supervised experiment. It does not alter `ofc_final_experiment`, the pending review templates, the original adapter, or the original GGUF.

The ZIP contains executable source-labeling, masked training, evaluation, export, and diagnostic code. It is NOT an already human-verified dataset, and it contains no model weights. Labels are generated locally when you run it.

## Run the full source-diverse experiment in Windows CMD

Download `OTSE_Automated_Source_QLoRA_v1.zip` to Downloads, then run:

```bat
cd /d "C:\Users\dipto\Downloads\Projects\Imran OFC 2026\OTSE-Optical-Transceiver-Spec-Extractor-imran"
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%USERPROFILE%\Downloads\OTSE_Automated_Source_QLoRA_v1.zip' -DestinationPath '.' -Force"
RUN_OTSE_AUTO_ALL.cmd
```

The archive root has `RUN_OTSE_AUTO.cmd` and `otse_auto\`. Those must sit beside your project's `data\`, `.venv\`, and `training_outputs\` folders. There must not be an extra outer package directory between the repository and `otse_auto`.

Do not run the old `ofc_final_experiment.02_build_reviewed_dataset`, `03_check_tokens`, or `04_train_safe` for this new experiment. They will still require the old human approval flags. This package uses its own label provenance and trainer.

The recommended full-source launcher processes all 399 eligible training PDFs and all 60 development PDFs in `training_outputs/otse-auto-source-all-v1/`. The standard `RUN_OTSE_AUTO.cmd` is a smaller 100-train/20-dev starting run. Both require no manual annotations.

## What the workflow does

1. Runs the supplied safety tests, verifies the live 48-field schema, hashes all 510 train/dev PDF sources, fingerprints the original adapter and deployment GGUF, and records package versions and GPU information.
2. In the standard run, selects 100 eligible training documents and 20 development documents; in the recommended full run, selects all eligible sources. It uses a deterministic technology/form-factor and recorded-family-diverse ordering. The original 43 internal-validation document IDs and all their recorded families are excluded from new training. On the supplied manifest this leaves 399 eligible training documents, not 450. No test/reserve document is labeled or trained on.
3. Reads native text across all PDF pages, creates stable page/block references, and retrieves group-relevant evidence. The actual Qwen tokenizer fits the source view before labeling. It starts with a 7,000-character cap, not a requirement to fill that cap.
4. Uses the existing local `qwen3:14b-otse8k` Ollama model for an evidence-only extraction, a second blinded extraction, and a skeptical check of agreed proposals. The second extraction does not see the first extraction or the old teacher suggestions. By default the SAME MODEL performs both roles, so these are not independent annotators.
5. Requires source-visible quote/block matches, schema correctness, numeric/string support, and conservative field-specific checks. Conflicting and unresolved fields are masked out. Absence means only absence in the exact selected input, not absence in the entire PDF.
6. Builds training/development JSONL, provenance, masks, and coverage reports. It automatically expands to additional eligible documents if the initial pool has insufficient coverage. It never converts ambiguous or pending fields into approved null targets merely to get training running.
7. Unloads the labeler models, evaluates the preserved adapter in HF/PEFT NF4, continues that adapter with QLoRA, and evaluates the candidate with identical evidence and generation settings in a separate process.
8. Compares precision, recall, micro-F1, field-level scores, exact-match recoveries/regressions, coverage, and paired document bootstrap intervals against the fixed machine reference. The candidate must improve this internal micro-F1 by at least one percentage point, lose no more than two precision points, and introduce no extra JSON-key failures to pass the internal gate. This is an engineering screening rule, not a publication significance test.
9. If that gate passes and existing llama.cpp conversion tools are located, exports a separate candidate Q4_K_M GGUF. If an existing llama-server is also located, evaluates baseline and candidate GGUFs with the same evidence and constrained JSON settings. Your baseline is NEVER overwritten or automatically promoted away.
10. Creates `OTSE_AUTO_RESULTS.zip` for the result/diagnostic handoff.

The application runs locally and stays in your terminal. It does not schedule background work. Keep the laptop connected to power and prevent sleep during a run. Label generation may take substantially longer than fine-tuning. The console reports progress and preserves completed requests; no duration estimate has been measured on your laptop.

## Initial settings

| Setting | Default |
|---|---|
| Preserved adapter | `training_outputs/qwen3-4b-otse-expanded-745/adapter_final` |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Labeler / checker | Existing `qwen3:14b-otse8k` for both roles |
| Reviewer context | 8,192 tokens, with request preflight and reserve |
| Evidence cap | 7,000 characters per group; automatically lowered before labeling when needed |
| Total training sequence | At most 4,096 actual Qwen tokens |
| Training | One corrective epoch, learning rate 0.00001, batch 1, accumulation 8 |
| Quantization | NF4, double quantization, BF16 when supported |
| Gradient checkpointing | Enabled; non-reentrant |
| Optimization | Existing adapter rank/alpha/dropout retained; no new adapter is silently created |
| Loss | Supervised field spans only; prompt, padding, and unresolved fields ignored |
| Default output | `training_outputs/otse-auto-source-v1/` |

These are conservative starting settings, not claimed optimal hyperparameters. The trainer checks actual token lengths and never silently truncates a target. The student is not asked to produce audit annotations at inference: it still emits the deployed field-group JSON.

## Partial-group supervision: the important change

The old builder rejected an entire group when any field was ambiguous. This workflow keeps the deployed full-group structure but masks uncertain field segments with loss label `-100`. Accepted keys are serialized BEFORE unknown placeholder keys, so unknown placeholders cannot condition later supervised answers under teacher forcing. Unknown null placeholders are not taught as ground-truth absence. Complete groups can train the whole answer; partial groups train only accepted field segments.

This is a deliberate experimental change, not conventional fully supervised gold-label SFT. Do not feed these JSONLs to the old generic trainer: doing so would discard the masks and train the unknown placeholders as null. Use this package's `training.py` and `dataset.py` together.

Absence-only training rows are capped, and accepted null fields are subsampled relative to positive fields to reduce pressure toward blanket abstention. The dev reference is not subsampled for this training balance.

## Augmentation

A deterministic quarter of positive training groups can receive one whole-block-order variant. The source blocks and all quoted evidence remain intact, including their page IDs. No table-row permutations, invented product variants, fabricated numbers, or unsupported optical conversions are generated. The same labels are retained only because the exact evidence set is unchanged. Development examples are never augmented.

Hard-negative interface/connector distinctions are learned from accepted source checks, not synthetic product specifications. The existing source facts also provide preservation examples, without blindly replaying the old unverified pseudolabel file.

## Full source-diverse launcher

To process all 399 eligible training documents and all 60 development documents in a separate experiment, run:

```bat
RUN_OTSE_AUTO_ALL.cmd
```

This creates `training_outputs/otse-auto-source-all-v1/`. It is a larger local labeling job, not a promise of better F1. Do not run the standard and all-source jobs simultaneously on the same GPU. The all-source workflow uses the same held-out split rules, not an additional test set.

## Resume or collect results

Run the same launcher with the same arguments to resume. Completed source annotations and exact matching model-request caches are reused. Training resumes from completed epoch checkpoints, not necessarily from the exact interrupted batch. Checkpoints, run configuration, model digests, and dataset signatures prevent silently combining incompatible runs.

```bat
RUN_OTSE_AUTO.cmd
MAKE_OTSE_AUTO_REPORT.cmd
```

For the all-source experiment:

```bat
MAKE_OTSE_AUTO_REPORT.cmd --output "training_outputs\otse-auto-source-all-v1"
```

The reports ZIP excludes model weights, raw PDFs, complete training data, and complete labeler conversations. It includes diagnostics, source IDs, environment versions, field-level score reports, and file paths. The local output folder retains complete provenance separately.

## Output files

```text
training_outputs/otse-auto-source-v1/
  run_config.json
  baseline_fingerprint.json
  source_plan.json
  label_model_fingerprints.json
  prepared_inputs/train/ and dev/
  annotations/train/ and dev/
  raw_label_calls/
  dataset/train.jsonl
  dataset/dev.jsonl
  dataset/field_provenance.jsonl
  dataset/dataset_report.json
  training/checkpoint-.../
  adapter_candidate/
  predictions/baseline/ and candidate/
  reports/comparison.json
  reports/baseline_field_details.jsonl
  reports/candidate_field_details.jsonl
  reports/gguf_comparison.json       [when native GGUF evaluation runs]
  candidate-Q4_K_M.gguf              [when export succeeds]
  OTSE_AUTO_RESULTS.zip
```

Every accepted label is tagged `AUTOMATED_SOURCE_CONSENSUS_NOT_HUMAN_GOLD`, `human_verified: false`, and `EXACT_SELECTED_INPUT_ONLY`. The old review files remain unchanged. This intentionally does not impersonate a human reviewer or retrofit human approval flags.

## GGUF tools not located automatically

The program searches common existing llama.cpp directories and PATH. It does not download or build new executables without your involvement. A missing converter does NOT invalidate completed labels, training, or HF comparison.

Supply your existing llama.cpp path when necessary:

```bat
EXPORT_OTSE_AUTO.cmd --llama-cpp-dir "C:\path\to\llama.cpp"
EVALUATE_OTSE_AUTO_GGUF.cmd --llama-cpp-dir "C:\path\to\llama.cpp"
```

CPU merging requires substantial RAM and disk. The conservative preflight asks for 24 GiB free disk and 16 GiB currently available RAM; these are safeguards, not a guarantee of the peak memory needed by every installed library version. Original files remain untouched if export fails. No additional GPU is required for CPU merging. Native GGUF comparison launches and stops only its own localhost llama-server child processes.

## Failure handling

- **Missing local labeler:** the program lists installed Ollama names. Use an already installed model through `--teacher` and optionally `--verifier`; it never silently downloads a multi-gigabyte model. The supplied default is the teacher named in your handoff.
- **Insufficient automated label coverage:** more eligible sources are processed automatically. If all eligible sources still fail the safeguards, the program stops with a coverage report rather than fabricating labels or asking you to approve thousands of fields.
- **Token budget:** evidence is reduced BEFORE labels are generated. Oversized completed targets are excluded, not truncated. True Qwen token counts are measured on your laptop.
- **GPU memory:** existing unrelated GPU inference servers can consume memory. The workflow unloads only the Ollama models it used and never kills unrelated user processes. A real CUDA/dependency failure is reported separately from annotation status.
- **Changed settings:** use a new `--output` directory. The old run is kept intact.
- **Candidate underperforms:** the baseline is retained. A failed experimental hypothesis is not a reason to manipulate labels, thresholds, or test references.

## Publication boundary

The generated development reference is machine-produced, filtered, incomplete, and tied to the selected input. Agreement between prompts using the same model does not establish independent factual correctness. A real number can appear in the wrong table row; lexical support is necessary but not sufficient. The skeptical pass and semantic checks reduce some risks but cannot remove all correlated errors.

Therefore `reports/comparison.json` and `reports/gguf_comparison.json` are AUTOMATED-REFERENCE DEVELOPMENT PROXIES, not the historical 60-document whole-PDF benchmark and not a new independent gold evaluation. Their coverage is reported explicitly. Bootstrap intervals measure sampling uncertainty only, not labeling bias. Do not claim 60+ factual Full60 F1 from these proxy scores. The historical exposed test set is not run or changed by this package; any later use needs the previously disclosed exploratory protocol.

No manual verification is required to run this experiment. Nevertheless, an automation-only experiment cannot supply independent human assurance for a paper. State this limitation honestly. No F1 improvement is promised.

## Validation performed before delivery

See `otse_auto/validation/VALIDATION_REPORT.json` and the unit-test log. Validation includes offline code/safety tests, type-schema alignment, source hashes, and source-block checks on genuine supplied PDFs. The actual Qwen tokenizer, live local Ollama responses, RTX 3500 training, Windows CMD launchers, and llama.cpp conversion were not executed in the authoring environment. Those checks run on your laptop; they are not claimed as already passed.

## Technical references

Official implementation references consulted for this package:

- Hugging Face PEFT quantization guide: https://huggingface.co/docs/peft/developer_guides/quantization
- Qwen model card: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Qwen3 masked-loss documentation: https://huggingface.co/docs/transformers/model_doc/qwen3
- Hugging Face fast tokenizer offset mapping: https://huggingface.co/docs/transformers/main_classes/tokenizer
- Ollama native structured chat API: https://docs.ollama.com/api/chat
- llama.cpp conversion and server documentation: https://github.com/ggml-org/llama.cpp

No unrelated library upgrades or new inference providers are required by the package.
