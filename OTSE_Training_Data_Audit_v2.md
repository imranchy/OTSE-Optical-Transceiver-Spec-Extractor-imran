# OTSE training-data audit and corrective pilot v1

**Date:** 10 October 2026  
**Status:** Full intake/structural screening completed; focused semantic pilot completed. No model training or new model inference performed.  
**Source basis:** The five collector ZIPs, `collection_manifest.json`, current project source code, original training datasets, and the unchanged frozen Full60 value scorer supplied in this conversation.

## 1. Decision

The missing training/development materials are now present. No additional corpus upload is required to begin the bounded corrective experiment.

The new evidence supports corrective supervision as a worthwhile experiment: some field-role mistakes and omissions are present in the teaching targets themselves. That does not establish that SFT alone will improve whole-document extraction, or that a particular F1 threshold is attainable. The first local run should measure the unchanged GGUF on identical reviewed pilot inputs before any adapter update.

The full 1,000-plus-record collection has received automated screening, **not complete semantic source adjudication**. The source-reviewed output in this package is a deliberately small pilot, not a replacement “verified gold” corpus.

## 2. Intake and inventory

All **1,360 archive entries** matched their recorded SHA-256 and byte count. The archives contain **450 training PDFs and 60 development PDFs**. Native text was extracted from **2,752 pages**. Fifteen pages have fewer than 80 stripped native-text characters; this is a review flag, not evidence that those pages contain missing text or require OCR.

| Material | Verified count | Interpretation |
|---|---:|---|
| Collected files | 1,360 | No integrity mismatches found |
| Training source PDFs | 450 | Current manifest train split |
| Development source PDFs | 60 | Current manifest dev split |
| JSONL files | 16 | Several overlapping dataset versions |
| Message-format datasets | 14 | Other two are index/unverified drafts |
| Message rows across versions | 1,375 | Not independent examples |
| Distinct serialized message payloads | 1,011 | Exact-message deduplication only |
| Main training rows | 745 | 391 distinct train documents |
| Internal-validation rows | 78 | 43 other train documents |

The corpus manifest contains 616 unique document hashes: 450 train, 60 dev, 60 test, 46 reserve. These official splits have no overlapping recorded family IDs. That validates the supplied grouping, not an independent audit of whether related product families were grouped adequately.

Older JSONL files overlap substantially. Do not concatenate every file named “train” or “gold.” Five of 15 legacy documents map to current train PDFs by hash; ten lack a current-manifest match and are excluded from this pilot. The 616-document index and `all_616_drafts.jsonl` span all splits and are not an approved training/reference dataset.

Detailed artifacts: `audit/archive_integrity.json`, `pdf_inventory.json`, `dataset_inventory.json`, `record_audit.jsonl`, and `duplicate_message_sets.json`.

## 3. Main training-data findings

### 3.1 No direct document leakage between the main 745/78 subsets

All 823 examples map to the current **training** partition. There is no document-ID overlap between the 745 examples and 78 internal-validation examples. Neither subset maps to current test or development documents.

However, **five recorded families occur in both internal subsets**:

```
group_11099930f3957a39
group_1b66dd82628913f7
group_91690c2f2ff5b634
group_a296009788c29bee
group_d87b881bd1c1fce0
```

This is not test-set contamination. It means the old internal validation was document-disjoint but not fully family-disjoint according to its own metadata. `audit/internal_family_overlap.json` lists the affected document IDs. The new pilot train/dev sets do not share recorded families.

### 3.2 Role confusion exists in the labels

A conservative screen flagged **138 values** assigned to host/media-interface fields despite being connectors, module form factors, or management labels: **121 in training and 17 in internal validation**, involving 94 documents.

These are heuristic flags for source review, not 138 independently adjudicated mistakes. Across all primary screening categories, 105 examples have at least one flag. Literal matching is only a screen: a unit conversion or accepted alias can be correct without a literal text match, and a literal number match can still belong to the wrong row.

Source review confirmed representative errors:

| Training source and original row | Original target | Reviewed page evidence / pilot treatment |
|---|---|---|
| `lq-lw100-lr4c.pdf`, train row 541 | Host `QSFP28`; media `QSFP28` | Electrical `CAUI-4`; optical `100GBASE-LR4`; form factor `QSFP28`; connector `LC` |
| `ftl4c1ql2c-transceiver-ds.pdf`, train row 110 | Host `QSFP+` | The first page explicitly names the `XLPPI` electrical interface |
| `so-qsfp28-dxx-a-r5.2.pdf`, train rows 49 and 94 | All interfaces/spectrum and optical fields null | The supplied page explicitly contains `CAUI-4`, `KP4`, `53.125 GBd`, TX -7 to +4 dBm and other supported values |

These are newly reviewed train-split examples, not copied test corrections. See the page images, exact source text and selected targets in `SOURCE_REVIEW.html`.

### 3.3 Empty targets need source review, not automatic acceptance or rejection

The main datasets contain **82 training targets and 11 internal-validation targets with no nonempty values**. Some may be legitimate for a particular evidence scope. The Smartoptics example above shows that at least some are wrong or incomplete relative to the actual SFT input.

It would be unsafe to approve all empty targets automatically, and equally unsafe to declare every one erroneous. All original records remain unapproved for the *new corrective* training pool pending semantic review; their original files are unchanged. The newly scoped pilot rows are separate replacements only for the small requested-field tasks they define.

### 3.4 The teacher, SFT and deployment evidence scopes differ

All 823 primary target objects equal the packaged teacher target objects. There is no evidence in these targets of a separate source-corrected answer layer.

For **685/745 training examples** and **66/78 internal-validation examples**, the exported SFT source text is longer than the evidence recorded for the teacher's last attempt. The supplied generation/export code is consistent with bounded teacher evidence but a later export of up to 30,000 source characters for SFT.

Separately, **709/745 training source excerpts exceed the deployment runner's 4,000-character source limit**; 51 reach 30,000 characters. Character counts are not token counts. The training tokenizer's actual truncation behavior must be measured rather than inferred from these lengths.

Length differences do not prove that every answer is wrong. They establish a supervision/input mismatch: a target may be incomplete for the longer SFT evidence, while deployment may omit information that training exposed. New examples should align the supplied evidence and target scope explicitly; merely adding epochs will not repair omitted deployment pages.

### 3.5 Current-schema violations

Nine primary targets fail the supplied current schema validator: **eight training and one internal-validation example**. Examples include scalar frequency/wavelength values where list-or-null is required. These are schema findings, not source-correctness judgments. `audit/schema_issues.json` retains the exact record and error.

## 4. Source ambiguity handling

The pilot does not guess “corrections” for contradictory source documents.

For `ds-td8004-sc4c-so-qsfp-dd800-800g-2xlr4-r6.4.pdf`, the narrative names `800GAUI-8` while the table names `800GAUI-4`. The pilot excludes the affected host field and frame instead of preferring one answer or labeling it null.

For `so-qsfp56-2c-2.pdf`, a wavelength interval is printed in reversed order and the LOS-assert row prints a positive 17 dBm. The affected range and LOS-assert fields are excluded. No guessed minus sign or endpoint correction is introduced.

When a single page lists multiple operating-temperature grades, scalar temperature fields are excluded unless a single grade is selected. Monitoring limits, damage thresholds and OMA power are distinguished from active average TX power and normal RX overload when the source supports that distinction.

These exclusions apply only to new pilot cases. They do not change the frozen Full60 references, aliases or exclusions.

## 5. Deliverable: source-reviewed pilot, not a final corpus

| Subset | Cases | Documents | Selected field decisions |
|---|---:|---:|---:|
| Corrective training | 35 | 12 train PDFs | 167 |
| Development comparison | 18 | 6 dev PDFs | 82 |
| Total | 53 | 18 PDFs | 249 |

Each case includes the exact reviewed first-page text, requested field definitions, a constrained JSON schema, target values, source hash, document ID, recorded family ID, review notes and any explicit ambiguity exclusions. Every target page was rendered and visually inspected.

Only selected fields on those 18 first pages were semantically reviewed. No claim is made that every field on each page, every page in those PDFs, or every original training record has been validated. This is AI-assisted review and remains fallible; it is not independent human certification.

Training requests only the reviewed keys. It does not teach every unreviewed field to be null. This changes the immediate task into requested-field extraction on bounded evidence, so a candidate must also pass whole-document and full-48-field regression tests before deployment.

The development subset was selected for diagnostic coverage. It is not random, representative or a fresh final test. Its F1 must not be compared numerically with the full-document Full60 F1. A 75% score on this small pilot would **not** establish 75% OTSE benchmark performance.

## 6. Code safety and what remains unimplemented

The old `training/continue_qlora.py` saves an updated adapter and then renames/replaces its input `adapter_final`. The new optional training runner never does that: it fingerprints the old adapter, refuses existing or overlapping output paths, and writes a distinct candidate directory.

The training script uses the existing local API style, a one-epoch NF4/BF16 continuation, learning rate 1e-5, batch 1, accumulation 4, and a 4,096-token bound. These are proposed pilot settings, not experimentally validated optimal settings. Token checks fail rather than silently truncate evidence. Training requires a successfully recorded matched-input pilot baseline first.

The current general GGUF runner still uses `source[:max_chars]` and starts its document timer after parsing. This package does **not** implement the full all-page retrieval upgrade or a new end-to-end timer. The pilot uses prepared page text and reports inference request times only. A broader evidence-aware pipeline remains necessary before a whole-document improvement claim.

No candidate merge, GGUF conversion, model hosting or deployment is performed. No teacher, student or GGUF model inference has been executed in this audit environment.

## 7. Testing and immediate action

**28 CPU unit/structural tests passed.** Tests cover real pilot PDF hashes, dataset shape, split/family separation, unreviewed/ambiguous-target rejection, output protection, token-overflow rejection using a mock tokenizer, and unchanged scoring semantics. Synthetic oracle predictions validate the scorer, not a model. All 18 actual pilot source hashes passed structural preflight.

Not tested here: the user's cached HF tokenizer, Windows CMD execution, installed GPU-training dependency compatibility, RTX 3500 memory fit, real GGUF inference, or training. Those limitations are retained in machine-readable provenance.

Extract `OTSE_Corrective_Pilot_v1.zip` into the project root. First run `RUN_OTSE_CORRECTIVE_PREFLIGHT.cmd`. After it passes, run the baseline GGUF server and `RUN_OTSE_CORRECTIVE_PILOT_BASELINE.cmd`. Upload the preflight JSON and baseline result folder. Do not launch an unreviewed large retraining job or replace deployment files.

The optional training launcher is provided for the subsequent bounded experiment. New whole-document scores are unknown. The frozen 53.68% GGUF micro-F1 and the existing reference package remain unchanged.
