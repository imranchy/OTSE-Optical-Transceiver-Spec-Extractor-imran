"""One-command OTSE automated labeling -> masked QLoRA -> matched dev comparison."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from .core import (BASE_ADAPTER, BASE_MODEL, DEFAULT_OUT, GROUPS, HERE, baseline_fingerprint, check_project_schema,
                   digest_obj, diverse_order, eligible_rows, project_root, read_json, source_path,
                   verify_baseline, write_json)
from . import __version__


def input_report(root, out):
    train, dev, protection = eligible_rows(root)
    schema = check_project_schema(root)
    # Hash every training/dev source, including preserved internal validation, once.
    from .core import load_manifest
    checked = []
    for row in load_manifest(root):
        if row['split'] in ('train', 'dev'):
            source_path(root, row)
            checked.append({'document_id': row['document_id'], 'sha256': row['sha256'], 'split': row['split']})
    report = {'schema': schema, 'source_hashes_checked': len(checked), 'sources': checked,
              'protection': protection, 'manual_approval_required': False,
              'note': 'This checks inputs and split eligibility, NOT factual correctness of annotations.'}
    write_json(out / 'reports/input_validation.json', report)
    print(f'Inputs: {len(checked)} source PDFs hashed; {len(train)} eligible training PDFs; {len(dev)} development PDFs.', flush=True)
    return report


def preflight(root, out, config):
    missing = [m for m in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'pydantic', 'fitz')
               if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError('Required packages absent in this interpreter: ' + ', '.join(missing) + '. Activate the existing working .venv. No automatic upgrades are performed.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in this Python interpreter. Activate the existing QLoRA .venv.')
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'pydantic', 'PyMuPDF'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unknown'
    fingerprint_path = out / 'baseline_fingerprint.json'
    if fingerprint_path.exists():
        verify_baseline(root, read_json(fingerprint_path))
    else:
        write_json(fingerprint_path, baseline_fingerprint(root))
    props = torch.cuda.get_device_properties(0)
    write_json(out / 'reports/environment.json', {'versions': versions, 'python': sys.version,
               'gpu': props.name, 'vram_bytes': props.total_memory, 'bf16_supported': torch.cuda.is_bf16_supported(),
               'base_model': BASE_MODEL, 'baseline_adapter': BASE_ADAPTER, 'run_configuration': config})
    input_report(root, out)
    print('Environment ready. Original weights fingerprinted; no pending review flags will be edited.', flush=True)


def config_from_args(args):
    return {'version': __version__, 'teacher': args.teacher, 'verifier': args.verifier or args.teacher,
            'ollama_url': args.ollama_url, 'review_context': args.review_context, 'max_chars': args.max_chars,
            'max_length': args.max_length, 'train_documents_initial': args.train_docs, 'dev_documents_initial': args.dev_docs,
            'epochs': args.epochs, 'lr': args.lr, 'grad_accum': args.grad_accum,
            'automatic_source_expansion': True, 'independent_human_review': False,
            'code_fingerprint': digest_obj({p.name: p.read_text(encoding='utf-8') for p in sorted(HERE.glob('*.py'))})}


def annotation_phase(root, out, config):
    from .training import get_tokenizer
    from .evidence import read_pages, fit_blocks, prompt_tokens
    from .labeling import Ollama, annotate, extraction_messages
    from .dataset import build
    tokenizer = get_tokenizer()
    client = Ollama(config['ollama_url'], out, tokenizer, config['review_context'])
    models = client.connect([config['teacher'], config['verifier']])
    signatures = out / 'label_model_fingerprints.json'
    if signatures.exists() and read_json(signatures) != models:
        raise RuntimeError('Local labeler model digest changed; choose a new --output directory to avoid mixed references')
    write_json(signatures, models)
    train, dev, _ = eligible_rows(root)
    pool = {'train': diverse_order(train), 'dev': diverse_order(dev)}
    limits = {'train': min(len(train), config['train_documents_initial'] or len(train)),
              'dev': min(len(dev), config['dev_documents_initial'] or len(dev))}
    if (out / 'source_plan.json').is_file():
        saved_limits = read_json(out / 'source_plan.json').get('limits', {})
        for split in limits:
            limits[split] = min(len(pool[split]), max(limits[split], saved_limits.get(split, 0)))
    consecutive_api_errors = 0
    started = time.perf_counter()
    while True:
        plan = {split: rows[:limits[split]] for split, rows in pool.items()}
        write_json(out / 'source_plan.json', {'rows': plan, 'limits': limits,
                   'selection': 'Deterministic technology/form-factor round robin; recorded families prioritized.'})
        total = sum(len(v) for v in plan.values()) * 3
        position = 0
        for split, rows in plan.items():
            for row in rows:
                needed = [g for g in GROUPS if not (out / 'annotations' / split / (row['document_id'] + '__' + g + '.json')).is_file()]
                if not needed:
                    position += 3
                    continue
                try:
                    pages = read_pages(root, row)
                except (ValueError, OSError) as exc:
                    write_json(out / 'excluded_sources' / (row['document_id'] + '.json'), {'source': row, 'reason': str(exc)})
                    position += 3
                    continue
                for group in GROUPS:
                    position += 1
                    destination = out / 'annotations' / split / (row['document_id'] + '__' + group + '.json')
                    if destination.is_file():
                        saved = read_json(destination)
                        if saved['sha256'] != row['sha256']:
                            raise RuntimeError('Cached annotation source changed')
                        continue
                    print(f'Labeling {position}/{total}: {split} {row["relative_path"]} / {group}', flush=True)
                    try:
                        cap = config['max_chars']
                        while True:
                            view = fit_blocks(pages, group, tokenizer, cap, config['max_length'])
                            largest = max(prompt_tokens(tokenizer, extraction_messages(group, view['blocks'], second)) for second in (False, True))
                            if largest + 3072 + 256 <= config['review_context']:
                                break
                            cap -= 400
                            if cap < 1400:
                                raise ValueError('Source cannot fit reviewer context without losing essential block structure')
                        packet = {**row, 'field_group': group, **view, 'retrieval_is_label_blind': True}
                        write_json(out / 'prepared_inputs' / split / (row['document_id'] + '__' + group + '.json'), packet)
                        result = annotate(client, config['teacher'], config['verifier'], packet)
                        write_json(destination, result)
                        consecutive_api_errors = 0
                        counts = {s: sum(r['state'] == s for r in result['fields'].values()) for s in ('supported', 'absent_in_input', 'unresolved')}
                        print('  machine-checked:', counts, flush=True)
                    except RuntimeError as exc:
                        consecutive_api_errors += 1
                        write_json(out / 'excluded_groups' / (row['document_id'] + '__' + group + '.json'), {'reason': str(exc), 'source': row})
                        if consecutive_api_errors >= 3:
                            raise RuntimeError('Three consecutive local labeling failures. See label_errors; resume with the same command after fixing the server. No annotations are automatically approved.') from exc
                    except ValueError as exc:
                        write_json(out / 'excluded_groups' / (row['document_id'] + '__' + group + '.json'), {'reason': str(exc), 'source': row})
        report = build(out, tokenizer, config['max_length'])
        print('Dataset coverage:', json.dumps(report['splits'], indent=2), flush=True)
        if report['ready_for_training']:
            client.unload_used_models([config['teacher'], config['verifier']])
            write_json(out / 'reports/labeling_runtime.json', {'wall_seconds_this_invocation': time.perf_counter() - started,
                'cached_calls_reused': True, 'human_verified': False})
            return report
        old_limits = dict(limits)
        for split, step in (('train', 100), ('dev', 20)):
            limits[split] = min(len(pool[split]), limits[split] + step)
        if limits == old_limits:
            client.unload_used_models([config['teacher'], config['verifier']])
            raise RuntimeError('All eligible sources were processed, but reliable automated-label coverage is below the minimum. See dataset_report.json. No manual approval is requested and no fabricated labels will be used.')
        print('Automatically expanding source review to', limits, '; no manual annotations needed.', flush=True)


def subprocess_stage(root, out, stage, extra=None):
    command = [sys.executable, '-m', 'otse_auto', stage, '--root', str(root), '--output', str(out)] + (extra or [])
    subprocess.run(command, cwd=str(root), check=True)


def make_report_zip(out):
    target = out / 'OTSE_AUTO_RESULTS.zip'
    include = []
    for subdir in ('reports', 'excluded_sources', 'excluded_groups', 'label_errors'):
        include.extend(p for p in (out / subdir).rglob('*') if p.is_file())
    include += [out / name for name in ('run_config.json', 'source_plan.json', 'training_complete.json',
                'last_error.json', 'baseline_fingerprint.json', 'label_model_fingerprints.json', 'dataset/dataset_report.json')
                if (out / name).is_file()]
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in include:
            z.write(p, str(p.relative_to(out)))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'check-inputs', 'label', 'train', 'infer', 'compare', 'export', 'gguf-eval', 'report', 'selftest'], nargs='?', default='run')
    parser.add_argument('--root')
    parser.add_argument('--output', default=DEFAULT_OUT)
    parser.add_argument('--teacher', default='qwen3:14b-otse8k')
    parser.add_argument('--verifier', default='')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--review-context', type=int, default=8192)
    parser.add_argument('--max-chars', type=int, default=7000)
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--train-docs', type=int, default=100, help='Initial source pool. 0=all eligible sources. Automatic expansion if coverage is too low.')
    parser.add_argument('--dev-docs', type=int, default=20, help='Initial dev source pool. 0=all 60. Automatic expansion if needed.')
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--which', choices=['baseline', 'candidate'], default='baseline')
    parser.add_argument('--llama-cpp-dir')
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--stop-after-dataset', action='store_true')
    args = parser.parse_args()
    if args.stage == 'selftest':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'), pattern='test_*.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    root = project_root(args.root)
    out = Path(args.output)
    out = out.resolve() if out.is_absolute() else (root / out).resolve()
    # Outputs cannot be the original adapter, GGUF directory, corpus, or package source.
    protected = [root / BASE_ADAPTER, root / 'data', root / 'deployment', HERE]
    if out == root or any(out == p.resolve() or p.resolve() in out.parents or out in p.resolve().parents for p in protected):
        raise ValueError('Output directory overlaps protected project files')
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / 'run_config.json'
    candidate = config_from_args(args)
    if min(args.max_chars, args.max_length, args.review_context, args.grad_accum) <= 0 or min(args.train_docs, args.dev_docs) < 0 or args.epochs <= 0 or args.lr <= 0:
        raise ValueError('Invalid nonpositive limits or training settings')
    if cfg_path.exists():
        config = read_json(cfg_path)
        if args.stage in ('run', 'label') and config != candidate:
            raise ValueError('Run settings/code differ from saved run. Use a new --output directory; existing data/checkpoints remain intact.')
    else:
        config = candidate
        write_json(cfg_path, config)
    try:
        if args.stage == 'check-inputs':
            input_report(root, out)
        elif args.stage in ('run', 'label'):
            preflight(root, out, config)
            annotation_phase(root, out, config)
            if args.stage == 'label' or args.stop_after_dataset:
                print('Automated masked dataset is ready. No manual review flags were required.', flush=True)
                return
            # Separate processes release all model/optimizer VRAM between stages.
            subprocess_stage(root, out, 'infer', ['--which', 'baseline'])
            subprocess_stage(root, out, 'train')
            subprocess_stage(root, out, 'infer', ['--which', 'candidate'])
            from .evaluation import compare
            comparison = compare(out)
            print(comparison['recommendation'], flush=True)
            if not args.no_export and comparison['candidate_passes_internal_gate']:
                # Merge/export may need substantial CPU RAM and disk. It is optional
                # and never allowed to invalidate completed labeling or training.
                from .export import find_tools
                tools = find_tools(root, args.llama_cpp_dir)
                if tools:
                    try:
                        subprocess_stage(root, out, 'export', ['--llama-cpp-dir', str(tools[0].parent)])
                        from .gguf_eval import server_binary
                        if server_binary(root, str(tools[0].parent)) and (out / 'candidate-Q4_K_M.gguf').is_file():
                            subprocess_stage(root, out, 'gguf-eval', ['--llama-cpp-dir', str(tools[0].parent)])
                    except subprocess.CalledProcessError:
                        print('GGUF export was deferred; candidate adapter and comparison report remain available.', flush=True)
                else:
                    write_json(out / 'reports/gguf_export.json', {'exported': False, 'reason': 'Existing llama.cpp tools were not located. Use the included EXPORT command with --llama-cpp-dir.'})
            verify_baseline(root, read_json(out / 'baseline_fingerprint.json'))
            print('\nDONE. Read:', out / 'reports/comparison.json', flush=True)
            print('Share results ZIP:', make_report_zip(out), flush=True)
        elif args.stage == 'train':
            from .training import train
            train(root, out, config)
        elif args.stage == 'infer':
            from .evaluation import predict
            predict(root, out, config, args.which)
        elif args.stage == 'compare':
            from .evaluation import compare
            print(json.dumps(compare(out), indent=2))
        elif args.stage == 'export':
            from .export import export_candidate
            export_candidate(root, out, args.llama_cpp_dir)
        elif args.stage == 'gguf-eval':
            from .gguf_eval import evaluate
            print(json.dumps(evaluate(root, out, args.llama_cpp_dir), indent=2))
        elif args.stage == 'report':
            print(make_report_zip(out))
    except Exception as exc:
        write_json(out / 'last_error.json', {'stage': args.stage, 'exception': type(exc).__name__, 'message': str(exc),
                   'traceback': traceback.format_exc(), 'manual_review_requested': False})
        print('\nSTOP:', exc, file=sys.stderr)
        print('Completed labels/checkpoints are retained. Diagnostic ZIP:', make_report_zip(out), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
