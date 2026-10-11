"""Matched automatic-reference dev comparison. Never call these scores Full60."""
from __future__ import annotations
import json
import time
import random
from collections import defaultdict
from pathlib import Path
from .core import BASE_ADAPTER, digest_obj, read_json, read_jsonl, write_json, write_jsonl
from . import scoring


def parse_object(content):
    # Accept only a complete response JSON object, not a guessed inner substring.
    try:
        value = json.loads(content.strip(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


def predict(root, out, config, which):
    import torch
    from .training import get_tokenizer, load_adapter
    rows = read_jsonl(out / 'dataset/dev.jsonl')
    if not rows:
        raise RuntimeError('Automated dev reference dataset missing')
    adapter = root / BASE_ADAPTER if which == 'baseline' else out / 'adapter_candidate'
    if not adapter.is_dir():
        raise FileNotFoundError(str(adapter))
    tokenizer = get_tokenizer()
    fingerprint = {str(p.name): __import__('hashlib').sha256(p.read_bytes()).hexdigest()
                   for p in adapter.glob('adapter*') if p.is_file()}
    import importlib.metadata
    software = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft')}
    settings = {'software_versions': software, 'adapter_fingerprint': fingerprint, 'temperature': 0, 'do_sample': False,
                'max_new_tokens': 1024, 'runtime': 'HF_PEFT_NF4', 'evidence': 'exact_training_retriever_dev_view'}
    folder = out / 'predictions' / which
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in rows:
        path = folder / (row['document_id'] + '__' + row['field_group'] + '.json')
        key = digest_obj([settings, row['source_view_sha256']])
        if not path.is_file() or read_json(path).get('prediction_key') != key:
            pending.append((row, path, key))
    if not pending:
        print(which + ' development inference already complete.', flush=True)
        return
    model, _ = load_adapter(root, adapter, trainable=False)
    model.eval()
    for index, (row, path, key) in enumerate(pending, 1):
        start = time.perf_counter()
        text = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt', truncation=False).to('cuda')
        count = inputs['input_ids'].shape[-1]
        # max_length limits SFT input+target, not the separately reserved inference output.
        # Keep the same prompt/evidence bytes; never truncate or change the source view.
        context_limit = int(config.get('review_context', 8192))
        if count + 1024 > context_limit:
            raise ValueError(f'Inference prompt plus 1024 output tokens ({count + 1024}) exceeds context limit {context_limit}')
        torch.cuda.synchronize()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=1024,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        tokens = output[0, count:]
        content = tokenizer.decode(tokens, skip_special_tokens=True)
        parsed = parse_object(content)
        expected = set(json.loads(row['messages'][-1]['content']))
        schema_ok = parsed is not None and set(parsed) == expected
        write_json(path, {'document_id': row['document_id'], 'field_group': row['field_group'], 'split': 'dev',
            'prediction_key': key, 'settings': settings, 'source_view_sha256': row['source_view_sha256'],
            'content': content, 'parsed': parsed, 'schema_keys_ok': schema_ok,
            'prompt_tokens': count, 'output_tokens': len(tokens),
            'possible_length_cutoff': len(tokens) >= 1024 and int(tokens[-1]) != tokenizer.eos_token_id,
            'generation_seconds': time.perf_counter() - generation_start,
            'request_wall_seconds': time.perf_counter() - start})
        print(f'{which} dev {index}/{len(pending)} {row["document_id"]}/{row["field_group"]}', flush=True)


def model_score(rows, folder):
    tallies, by_field, by_doc = [], defaultdict(list), defaultdict(list)
    exact = {}
    elapsed, malformed = [], 0
    field_rows = []
    for row in rows:
        path = folder / (row['document_id'] + '__' + row['field_group'] + '.json')
        pred = read_json(path)
        if pred['source_view_sha256'] != row['source_view_sha256'] or pred['split'] != 'dev':
            raise ValueError('Reference/prediction evidence or split mismatch')
        parsed = pred.get('parsed')
        malformed += not pred.get('schema_keys_ok', False)
        elapsed.append(pred.get('request_wall_seconds', 0))
        for field in row['known_fields']:
            value = parsed.get(field, scoring.MISSING) if isinstance(parsed, dict) else scoring.MISSING
            detail = scoring.score_one({'field': field, 'gold_value': row['target_values'][field]}, value)
            counts = detail['counts']
            tallies.append(counts)
            by_field[field].append(counts)
            by_doc[row['document_id']].append(counts)
            key = row['document_id'] + '/' + field
            exact[key] = detail['exact_match']
            field_rows.append({'document_id': row['document_id'], 'field': field,
                               'reference_state': row['target_states'][field], **detail})
    total = scoring.metrics(scoring.add_counts(tallies))
    per_field = {f: scoring.metrics(scoring.add_counts(c)) for f, c in by_field.items()}
    finite = [m['micro_f1'] for m in per_field.values() if m['micro_f1'] is not None]
    total.update({'macro_field_f1': sum(finite) / len(finite) if finite else None,
                  'schema_key_failure_groups': malformed, 'inference_request_seconds_total': sum(elapsed),
                  'latency_scope': 'Tokenization plus GPU generation; excludes label creation/model loading/PDF retrieval.'})
    return total, per_field, {d: scoring.add_counts(v) for d, v in by_doc.items()}, exact, field_rows


def bootstrap_delta(a, b, reps=2000):
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    rng = random.Random(42)
    delta = []
    for _ in range(reps):
        selected = [rng.choice(ids) for _ in ids]
        am = scoring.metrics(scoring.add_counts(a[i] for i in selected))['micro_f1']
        bm = scoring.metrics(scoring.add_counts(b[i] for i in selected))['micro_f1']
        if am is not None and bm is not None:
            delta.append(bm - am)
    delta.sort()
    if not delta:
        return None
    return {'lower_95': delta[int(0.025 * (len(delta) - 1))],
            'upper_95': delta[int(0.975 * (len(delta) - 1))], 'resamples': reps,
            'unit': 'document', 'warning': 'Sampling uncertainty only; does not capture machine-label bias or family correlation.'}


def compare(out, prefix=""):
    rows = read_jsonl(out / 'dataset/dev.jsonl')
    base, bf, bd, be, br = model_score(rows, out / ('predictions/' + prefix + 'baseline'))
    cand, cf, cd, ce, cr = model_score(rows, out / ('predictions/' + prefix + 'candidate'))
    metrics_available = all(x is not None for x in (base['micro_f1'], cand['micro_f1'], base['precision'], cand['precision']))
    accepted = bool(metrics_available and cand['micro_f1'] >= base['micro_f1'] + 0.01
                    and cand['precision'] >= base['precision'] - 0.02
                    and cand['schema_key_failure_groups'] <= base['schema_key_failure_groups'])
    documents = {r['document_id'] for r in rows}
    known = sum(len(r['known_fields']) for r in rows)
    report = {'runtime_comparison': 'llama.cpp Q4_K_M' if prefix else 'HF PEFT NF4', 'scope': 'AUTOMATED_REFERENCE_DEV_PROXY_ONLY_NOT_FULL60', 'human_verified': False,
        'independent_test_claim': False, 'reference_scope': 'EXACT_SELECTED_INPUT_ONLY',
        'baseline': base, 'candidate': cand, 'baseline_by_field': bf, 'candidate_by_field': cf,
        'known_field_decisions': known, 'possible_field_decisions_on_evaluated_documents': 48 * len(documents),
        'field_decision_coverage': known / (48 * len(documents)) if documents else 0,
        'documents_evaluated': len(documents),
        'recovered_exact_fields': sum(not be[k] and ce[k] for k in be),
        'regressed_exact_fields': sum(be[k] and not ce[k] for k in be),
        'paired_document_bootstrap_delta_f1': bootstrap_delta(bd, cd),
        'selection_rule': 'At least +1 F1 percentage point, no more than 2 precision points lost, and no extra schema-key failures.',
        'candidate_passes_internal_gate': accepted,
        'baseline_deployment_overwritten': False,
        'recommendation': 'Candidate eligible for separate GGUF validation; not automatically promoted.' if accepted else 'Retain original baseline; candidate did not pass the internal gate.',
        'warning': 'References are model-generated and source-filtered. Correlated labeling errors and selection bias remain. Do not compare this score with historical Full60 or claim 60+ factual F1 from it.'}
    write_json(out / ('reports/' + prefix + 'comparison.json'), report)
    write_jsonl(out / ('reports/' + prefix + 'baseline_field_details.jsonl'), br)
    write_jsonl(out / ('reports/' + prefix + 'candidate_field_details.jsonl'), cr)
    return report
