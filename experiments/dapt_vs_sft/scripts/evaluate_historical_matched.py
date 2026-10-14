"""OTSE frozen historical-protocol comparison (15 PDFs / 45 tasks / 520 field decisions).

Use inside the original OTSE repo, with otse_auto/ installed. This script NEVER
changes references or trains weights. Historical reference labels are automatic,
not independent human gold. The historical 80.70% student anchor was GGUF, while
these model arms are HF NF4; compare those configurations with care.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

BASE = 'Qwen/Qwen3-4B-Instruct-2507'
ADAPTERS = {
    'sft': 'training_outputs/qwen3-4b-otse-expanded-745/adapter_final',
    'dapt': 'experiments/dapt_vs_sft/outputs/dapt_only/adapter_final',
    'dapt_sft': 'experiments/dapt_vs_sft/outputs/dapt_then_sft/adapter_final',
}
MODELS = ('base', 'sft', 'dapt', 'dapt_sft')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')
    os.replace(temp, path)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def candidate_datasets(root):
    return sorted((root / 'training_outputs').glob('*/dataset/dev.jsonl'))


def validate_dataset(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    required = {'document_id', 'field_group', 'messages', 'known_fields', 'target_values', 'source_view_sha256', 'split'}
    for index, r in enumerate(rows):
        missing = required - set(r)
        if missing:
            raise ValueError(f'Row {index} missing {sorted(missing)}: {path}')
        if r['split'] != 'dev':
            raise ValueError(f'Non-development row found: {r["document_id"]}')
        if r['messages'][-1]['role'] != 'assistant':
            raise ValueError('Expected original prompt + assistant reference for each task')
        if set(r['known_fields']) != set(r['target_values']):
            raise ValueError('known_fields differs from target_values')
    docs = {r['document_id'] for r in rows}
    tasks = {(r['document_id'], r['field_group']) for r in rows}
    known = sum(len(r['known_fields']) for r in rows)
    if (len(rows), len(docs), len(tasks), known) != (45, 15, 45, 520):
        raise ValueError(f'Not the historical-sized reference: rows={len(rows)}, PDFs={len(docs)}, unique tasks={len(tasks)}, known field decisions={known}; expected 45, 15, 45, 520. Do not select a different dataset to force a score.')
    return rows


def choose_dataset(root, requested):
    if requested:
        path = Path(requested)
        if not path.is_absolute(): path = root / path
        return path.resolve(), validate_dataset(path.resolve())
    good, bad = [], []
    for p in candidate_datasets(root):
        try:
            good.append((p, validate_dataset(p)))
        except (ValueError, KeyError) as exc:
            bad.append((p, str(exc)))
    if len(good) != 1:
        desc = '\n'.join('  ' + str(p) for p, _ in good) or '  none'
        hints = '\n'.join('  ' + str(p) for p in candidate_datasets(root)) or '  no dataset/dev.jsonl candidates found'
        raise RuntimeError(f'Expected exactly one historical-shaped frozen dev dataset, found {len(good)}. Matching:\n{desc}\nAll candidates:\n{hints}\nUse --dataset PATH to specify the exact archived historical 45-task reference. Do not regenerate or handpick it.')
    return good[0]


def load_model(root, name):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for local QLoRA inference')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    m = AutoModelForCausalLM.from_pretrained(BASE, device_map={'': 0}, dtype=dtype, quantization_config=q, attn_implementation='sdpa')
    if name != 'base':
        adapter = root / ADAPTERS[name]
        cfg = load_json(adapter / 'adapter_config.json')
        if cfg.get('base_model_name_or_path', '').rstrip('/') != BASE:
            raise ValueError(f'Wrong adapter base model for {name}: {cfg.get("base_model_name_or_path")}')
        m = PeftModel.from_pretrained(m, str(adapter), is_trainable=False)
    m.eval()
    m.config.use_cache = True
    return m


def fingerprint(root, name):
    if name == 'base':
        return 'base|' + BASE
    p = root / ADAPTERS[name]
    cfg, weights = p / 'adapter_config.json', p / 'adapter_model.safetensors'
    if not cfg.is_file() or not weights.is_file():
        raise FileNotFoundError('Missing adapter: ' + str(p))
    return hash_file(cfg) + ':' + hash_file(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--dataset', default='', help='Frozen historical dev.jsonl; autodetect if unambiguous')
    parser.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    parser.add_argument('--output', default='experiments/dapt_vs_sft/results/historical_protocol')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    from otse_auto import scoring
    from otse_auto.evaluation import model_score, parse_object
    from transformers import AutoTokenizer
    import torch

    dataset_path, rows = choose_dataset(root, args.dataset)
    target = root / args.output
    target.mkdir(parents=True, exist_ok=True)
    dataset_hash = hash_file(dataset_path)
    # Keep the historical evaluation from silently becoming a training-set score.
    sft_train = root / 'data/corpus_v2/pseudo_qlora/train.jsonl'
    sft_val = root / 'data/corpus_v2/pseudo_qlora/validation.jsonl'
    doc_ids = {r['document_id'] for r in rows}
    overlap = {}
    for key, path in [('train', sft_train), ('validation', sft_val)]:
        if path.is_file():
            source_ids = {json.loads(line)['document_id'] for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()}
            overlap[key] = sorted(doc_ids & source_ids)
    if overlap.get('train'):
        raise ValueError(f'Historical evaluation overlaps original SFT training PDFs: {overlap["train"]}')
    if overlap.get('validation'):
        print(f'WARNING: {len(overlap["validation"])} historical PDFs overlap original SFT validation: development-only comparison', flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    # Old evaluation used the original messages and a maximum of 1024 generation tokens.
    inputs = []
    for row in rows:
        prompt = tok.apply_chat_template(row['messages'][:-1], tokenize=False, add_generation_prompt=True)
        tokens = tok(prompt, add_special_tokens=False)['input_ids']
        if len(tokens) + 1024 > 8192:
            raise ValueError(f'Overlength historical prompt for {row["document_id"]}/{row["field_group"]}')
        inputs.append((row, prompt))
    save_json(target / 'protocol.json', {
        'reference_path': str(dataset_path), 'reference_sha256': dataset_hash,
        'label_status': 'AUTOMATED_SILVER_REFERENCE_NOT_EXPERT_GOLD',
        'documents': 15, 'tasks': 45, 'known_field_decisions': 520,
        'original_sft_overlap': {k: len(v) for k, v in overlap.items()},
        'input': 'original frozen dev.jsonl messages[:-1] (no evidence rebuild)',
        'scoring': 'otse_auto.evaluation.model_score -> otse_auto.scoring.score_one, canonical default',
        'generation': {'max_new_tokens': 1024, 'do_sample': False, 'temperature': 0},
        'model_runtime': 'HF PEFT NF4; note original historical GGUF 80.70% uses different runtime',
        'test_status': 'historical automated-reference development proxy, not independent human gold',
        'model_paths': ADAPTERS,
    })

    summary = {}
    for name in args.models:
        model_hash = fingerprint(root, name)
        out = target / 'predictions' / name
        out.mkdir(parents=True, exist_ok=True)
        pending = []
        for row, prompt in inputs:
            path = out / f'{row["document_id"]}__{row["field_group"]}.json'
            if path.is_file():
                rec = load_json(path)
                if (rec.get('model_fingerprint') == model_hash and
                    rec.get('reference_sha256') == dataset_hash and
                    rec.get('source_view_sha256') == row['source_view_sha256'] and
                    rec.get('max_new_tokens') == 1024 and rec.get('split') == 'dev'):
                    continue
            pending.append((row, prompt, path))
        print(f'{name}: {len(pending)} requests remaining / 45', flush=True)
        model = None
        try:
            if pending: model = load_model(root, name)
            for i, (row, prompt, path) in enumerate(pending, 1):
                x = tok(prompt, add_special_tokens=False, return_tensors='pt', truncation=False).to('cuda')
                n = x['input_ids'].shape[1]
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    y = model.generate(**x, do_sample=False, max_new_tokens=1024,
                                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                generated = y[0,n:]
                content = tok.decode(generated, skip_special_tokens=True)
                parsed = parse_object(content)
                expected = set(json.loads(row['messages'][-1]['content']))
                save_json(path, {
                    'model': name, 'model_fingerprint': model_hash,
                    'reference_sha256': dataset_hash,
                    'document_id': row['document_id'], 'field_group': row['field_group'],
                    'split': 'dev', 'source_view_sha256': row['source_view_sha256'],
                    'max_new_tokens': 1024, 'content': content, 'parsed': parsed,
                    'schema_keys_ok': parsed is not None and set(parsed) == expected,
                    'prompt_tokens': n, 'output_tokens': len(generated),
                    'possible_length_cutoff': len(generated) >= 1024,
                    'request_wall_seconds': seconds,
                })
                print(f'  {name} {i}/{len(pending)} {row["document_id"]}/{row["field_group"]} {seconds:.1f}s', flush=True)
        finally:
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()
        overall, per_field, per_doc, exact, details = model_score(rows, out)
        save_json(target / f'{name}_metrics.json', {'model': name, 'reference_sha256': dataset_hash,
                  'automated_reference': True, 'metrics': overall,
                  'per_field': per_field, 'per_document_counts': per_doc})
        summary[name] = {'canonical_micro_f1': overall['micro_f1'],
                         'precision': overall['precision'], 'recall': overall['recall'],
                         'n': overall['n'], 'gold_present': overall['gold_present'],
                         'schema_key_failure_groups': overall['schema_key_failure_groups'],
                         'model_fingerprint': model_hash}
        save_json(target / 'comparison_summary.json', summary)
        print(name, 'F1:', overall['micro_f1'], 'n=', overall['n'], 'gold_present=', overall['gold_present'], flush=True)
    print('All requested models completed:', target, flush=True)

if __name__ == '__main__':
    main()
