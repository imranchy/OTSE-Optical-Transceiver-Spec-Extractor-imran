"""OTSE frozen historical 45-task GGUF/Ollama comparison + resource profiling.

Run from repository root:
  python experiments/dapt_vs_sft/scripts/evaluate_gguf_resources.py

Requires the earlier evaluate_historical_matched.py in this same scripts folder,
OTSE's original otse_auto package, transformers, and psutil.
Ollama must be running on localhost:11434. Never edits training labels/models.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean, median

MODEL_TAGS = {
    'base': 'qwen3:4b-instruct-2507-q4_K_M',
    'sft': 'otse-qwen4b-final:latest',
    'dapt': 'otse-qwen4b-dapt:latest',
    'dapt_sft': 'otse-qwen4b-dapt-sft:latest',
    'teacher': 'qwen3:14b-otse8k',
}
KNOWN_GGUFS = {
    'sft': 'deployment/otse-qwen4b-Q4_K_M.gguf',
    'dapt': 'deployment/otse-dapt-Q4_K_M.gguf',
    'dapt_sft': 'deployment/otse-dapt-sft-Q4_K_M.gguf',
}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ollama_post(url, payload, timeout=600):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def ollama_get(url):
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode('utf-8'))


def gguf_info(root, tag, name, api):
    show = ollama_post(api + '/api/show', {'model': tag}, timeout=60)
    tags = ollama_get(api + '/api/tags').get('models', [])
    match = next((m for m in tags if m.get('name') == tag or m.get('model') == tag or m.get('name') == tag.replace(':latest','')), {})
    modelfile = show.get('modelfile', '')
    blob = None
    m = re.search(r'^FROM\s+(.+)$', modelfile, flags=re.MULTILINE)
    if m:
        candidate = Path(m.group(1).strip().strip('"'))
        if candidate.is_file():
            blob = candidate
    local = root / KNOWN_GGUFS[name] if name in KNOWN_GGUFS else None
    if local and not local.is_file():
        raise FileNotFoundError('Expected GGUF missing: ' + str(local))
    local_size = local.stat().st_size if local else None
    blob_size = blob.stat().st_size if blob else None
    if local and blob and (local_size != blob_size or digest(local) != digest(blob)):
        raise ValueError(f'Ollama model {tag} does not match repository GGUF {local}')
    details = show.get('details') or match.get('details') or {}
    return {
        'tag': tag,
        'ollama_digest': match.get('digest'),
        'ollama_size_bytes': match.get('size'),
        'gguf_path': str(local) if local else (str(blob) if blob else None),
        'gguf_size_bytes': local_size if local else blob_size,
        'gguf_sha256': digest(local) if local else (digest(blob) if blob else None),
        'parameter_size': details.get('parameter_size'),
        'quantization_level': details.get('quantization_level'),
        'model_family': details.get('family'),
        'ollama_template': show.get('template'),
        'ollama_modelfile': modelfile,
    }


def gpu_read():
    exe = shutil.which('nvidia-smi')
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=6, check=True)
        first = result.stdout.strip().splitlines()[0].split(',')
        return {'gpu_util_pct': float(first[0].strip()),
                'gpu_memory_used_mib': float(first[1].strip()),
                'gpu_memory_total_mib': float(first[2].strip()),
                'gpu_power_w': float(first[3].strip()) if first[3].strip() not in ('[N/A]', 'N/A') else None}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class Monitor:
    """System-wide samples, not model-exclusive process attribution."""
    def __init__(self, psutil, interval):
        self.psutil = psutil
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _run(self):
        self.psutil.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            v = self.psutil.virtual_memory()
            sample = {
                'cpu_util_pct': self.psutil.cpu_percent(interval=None),
                'ram_used_gib': v.used / 1024**3,
                'ram_available_gib': v.available / 1024**3,
                'ram_util_pct': v.percent,
            }
            gpu = gpu_read()
            if gpu:
                sample.update(gpu)
            self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=8)

    def summary(self, baseline=None):
        result = {'sample_count': len(self.samples), 'sampling_interval_seconds': self.interval,
                  'scope': 'system-wide; other applications and Ollama model loading may contribute'}
        for key in ('cpu_util_pct', 'ram_used_gib', 'ram_available_gib', 'ram_util_pct',
                    'gpu_util_pct', 'gpu_memory_used_mib', 'gpu_memory_total_mib', 'gpu_power_w'):
            values = [s[key] for s in self.samples if s.get(key) is not None]
            result[key + '_mean'] = mean(values) if values else None
            result[key + '_peak'] = max(values) if values else None
        if baseline:
            for key in ('ram_used_gib', 'gpu_memory_used_mib'):
                peak = result.get(key + '_peak')
                base = baseline.get(key)
                result[key + '_peak_minus_idle'] = max(0, peak - base) if peak is not None and base is not None else None
        return result


def baseline_snapshot(psutil):
    v = psutil.virtual_memory()
    gpu = gpu_read() or {}
    return {'cpu_util_pct': psutil.cpu_percent(interval=0.1),
            'ram_used_gib': v.used / 1024**3,
            'ram_available_gib': v.available / 1024**3,
            'gpu_memory_used_mib': gpu.get('gpu_memory_used_mib'),
            'gpu_util_pct': gpu.get('gpu_util_pct')}


def csv_write(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--dataset', default='', help='Original frozen 45-task dev.jsonl (autodetected if unique)')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_TAGS), default=['teacher', 'sft', 'dapt', 'dapt_sft'])
    parser.add_argument('--output', default='experiments/dapt_vs_sft/results/gguf_final_9k')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--sample-interval', type=float, default=1.0)
    parser.add_argument('--request-timeout', type=int, default=1200)
    parser.add_argument('--num-ctx', type=int, default=9216)
    parser.add_argument('--max-generate', type=int, default=4096)
    parser.add_argument('--context-reserve', type=int, default=128)
    args = parser.parse_args()
    if args.num_ctx < 2048 or args.max_generate < 1024 or args.context_reserve < 0:
        parser.error('Invalid context/generation limits')
    if args.sample_interval < 0.2:
        parser.error('--sample-interval must be >= 0.2 seconds')
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'experiments/dapt_vs_sft/scripts'))
    try:
        import psutil
        from transformers import AutoTokenizer
        from otse_auto.evaluation import model_score, parse_object
        from evaluate_historical_matched import choose_dataset
    except ImportError as e:
        raise SystemExit(f'Missing local module/dependency: {e}. Keep evaluate_historical_matched.py alongside this script.')

    dataset_path, rows = choose_dataset(root, args.dataset)
    dataset_hash = digest(dataset_path)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    api = args.ollama_url.rstrip('/')
    try:
        ollama_get(api + '/api/tags')
    except Exception as e:
        raise SystemExit(f'Ollama not reachable at {api}: {e}')

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', use_fast=True)
    prompts = []
    for row in rows:
        prompt = tok.apply_chat_template(row['messages'][:-1], tokenize=False, add_generation_prompt=True)
        input_tokens = len(tok.encode(prompt, add_special_tokens=False))
        budget = min(args.max_generate, args.num_ctx - input_tokens - args.context_reserve)
        if budget < 1024:
            raise ValueError(f'Insufficient context for {row["document_id"]}/{row["field_group"]}: prompt={input_tokens}, available_output={budget}. Increase --num-ctx for ALL models.')
        prompts.append((row, prompt, input_tokens, budget))

    disk = shutil.disk_usage(root)
    machine = {
        'cpu_logical_cores': psutil.cpu_count(logical=True),
        'cpu_physical_cores': psutil.cpu_count(logical=False),
        'ram_total_gib': psutil.virtual_memory().total / 1024**3,
        'disk_total_gib': disk.total / 1024**3,
        'disk_free_gib_at_start': disk.free / 1024**3,
        'gpu': gpu_read(),
    }
    protocol = {
        'dataset_path': str(dataset_path), 'dataset_sha256': dataset_hash,
        'reference_type': 'historical automated silver labels; not independent expert gold',
        'documents': 15, 'tasks': 45, 'field_decisions': 520,
        'prompt': 'Qwen3-4B tokenizer chat template applied to frozen messages[:-1]',
        'ollama_raw': True,
        'raw_reason': 'Preformatted prompt sent verbatim; prevent a second Ollama chat template',
        'ollama_options': {'temperature': 0, 'num_ctx': args.num_ctx, 'num_predict': 'dynamic shared rule, up to ' + str(args.max_generate), 'seed': 42, 'think': False},
        'token_budget_rule': 'min(max_generate, num_ctx - Qwen3-4B rendered prompt token count - context_reserve); same per task for all models',
        'context_reserve': args.context_reserve,
        'prompt_token_counts': {'min': min(x[2] for x in prompts), 'max': max(x[2] for x in prompts)},
        'generation_budgets': {'min': min(x[3] for x in prompts), 'max': max(x[3] for x in prompts)},
        'scoring': 'Original otse_auto.evaluation.model_score canonical score_one',
        'latency_scope': 'Ollama API request wall time, including prompt processing and generation; excludes PDF retrieval',
        'resource_scope': 'System-wide sampling during each inference request; not exclusive model attribution',
        'machine': machine,
        'model_tags': {n: MODEL_TAGS[n] for n in args.models},
        'caveat': 'New 9K non-thinking protocol; not the original 8K/1024 historical configuration. All four models rerun; scores may differ.',
    }
    save_json(output / 'protocol.json', protocol)
    summary = {}
    for name in args.models:
        tag = MODEL_TAGS[name]
        info = gguf_info(root, tag, name, api)
        out = output / 'predictions' / name
        out.mkdir(parents=True, exist_ok=True)
        fingerprint = info.get('gguf_sha256') or info.get('ollama_digest')
        if not fingerprint:
            raise ValueError('Cannot fingerprint model: ' + tag)
        save_json(output / f'{name}_model_info.json', info)
        pending = []
        for row, prompt, input_tokens, budget in prompts:
            path = out / f'{row["document_id"]}__{row["field_group"]}.json'
            if path.exists():
                old = load_json(path)
                if (old.get('model_fingerprint') == fingerprint and
                    old.get('reference_sha256') == dataset_hash and
                    old.get('source_view_sha256') == row['source_view_sha256'] and
                    old.get('protocol_version') == 2 and old.get('num_ctx') == args.num_ctx and old.get('num_predict') == budget and old.get('think') is False and old.get('prompt_sha256') == hashlib.sha256(prompt.encode('utf-8')).hexdigest() and old.get('split') == 'dev'):
                    continue
            pending.append((row, prompt, input_tokens, budget, path))
        # Release all other Ollama models before profiling this model.
        try:
            running = ollama_get(api + '/api/ps').get('models', [])
            for active in running:
                active_tag = active.get('name') or active.get('model')
                if active_tag:
                    ollama_post(api + '/api/generate', {'model': active_tag, 'prompt': '', 'stream': False, 'keep_alive': 0, 'raw': True}, timeout=120)
        except Exception as e:
            print('WARNING: Could not fully unload resident Ollama models:', e, flush=True)
        print(f'{name} ({tag}): {len(pending)} remaining / {len(rows)}', flush=True)
        for i, (row, prompt, input_tokens, budget, path) in enumerate(pending, 1):
            # Load/warm the model outside the timed inference request.
            # Ollama's keep_alive reduces repeated model-loading overhead.
            if i == 1:
                ollama_post(api + '/api/generate', {'model': tag, 'prompt': '', 'raw': True,
                            'stream': False, 'think': False, 'keep_alive': '15m', 'options': {'num_ctx': args.num_ctx, 'num_predict': 1}},
                            timeout=args.request_timeout)
            baseline = baseline_snapshot(psutil)
            with Monitor(psutil, args.sample_interval) as mon:
                start = time.perf_counter()
                response = ollama_post(api + '/api/generate', {
                    'model': tag, 'prompt': prompt, 'raw': True, 'stream': False,
                    'keep_alive': '15m', 'think': False,
                    'options': {'temperature': 0, 'num_ctx': args.num_ctx, 'num_predict': budget, 'seed': 42},
                }, timeout=args.request_timeout)
                elapsed = time.perf_counter() - start
            if response.get('error'):
                raise RuntimeError(response['error'])
            content = response.get('response', '')
            parsed = parse_object(content)
            expected = set(json.loads(row['messages'][-1]['content']))
            resource = mon.summary(baseline)
            record = {
                'protocol_version': 2, 'model': name, 'model_tag': tag,
                'model_fingerprint': fingerprint, 'reference_sha256': dataset_hash,
                'document_id': row['document_id'], 'field_group': row['field_group'],
                'source_view_sha256': row['source_view_sha256'], 'split': 'dev',
                'content': content, 'parsed': parsed,
                'think': False, 'num_ctx': args.num_ctx, 'num_predict': budget,
                'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                'rendered_prompt_tokens_estimate': input_tokens,
                'ollama_thinking': response.get('thinking'),
                'ollama_done_reason': response.get('done_reason'),
                'schema_keys_ok': parsed is not None and set(parsed) == expected,
                'request_wall_seconds': elapsed,
                'ollama_prompt_eval_count': response.get('prompt_eval_count'),
                'ollama_eval_count': response.get('eval_count'),
                'ollama_prompt_eval_duration_ns': response.get('prompt_eval_duration'),
                'ollama_eval_duration_ns': response.get('eval_duration'),
                'ollama_load_duration_ns': response.get('load_duration'),
                'ollama_total_duration_ns': response.get('total_duration'),
                'resource': resource,
                'resource_baseline': baseline,
            }
            save_json(path, record)
            print(f'  {name} {i}/{len(pending)} {row["document_id"]}/{row["field_group"]} '
                  f'{elapsed:.1f}s budget={budget} done={response.get("done_reason")} '
                  f'VRAMpeak={resource.get("gpu_memory_used_mib_peak")}', flush=True)
        overall, by_field, by_doc, _, details = model_score(rows, out)
        records = [load_json(out / f'{r["document_id"]}__{r["field_group"]}.json') for r in rows]
        durations = [r['request_wall_seconds'] for r in records]
        samples = [r['resource'] for r in records]
        def maximum(key):
            vals = [r.get(key) for r in samples if r.get(key) is not None]
            return max(vals) if vals else None
        def average(key):
            vals = [r.get(key) for r in samples if r.get(key) is not None]
            return mean(vals) if vals else None
        toks = sum(r.get('ollama_eval_count') or 0 for r in records)
        gen_sec = sum((r.get('ollama_eval_duration_ns') or 0) / 1e9 for r in records)
        resource_summary = {
            'model_file_size_gb_decimal': info['gguf_size_bytes'] / 1e9 if info['gguf_size_bytes'] else None,
            'model_file_size_gib': info['gguf_size_bytes'] / 1024**3 if info['gguf_size_bytes'] else None,
            'total_request_wall_seconds': sum(durations),
            'mean_request_wall_seconds': mean(durations),
            'median_request_wall_seconds': median(durations),
            'min_request_wall_seconds': min(durations),
            'max_request_wall_seconds': max(durations),
            'generated_tokens': toks,
            'generation_tokens_per_second': toks / gen_sec if gen_sec else None,
            'system_cpu_util_peak_pct': maximum('cpu_util_pct_peak'),
            'system_cpu_util_mean_of_request_means_pct': average('cpu_util_pct_mean'),
            'system_ram_peak_gib': maximum('ram_used_gib_peak'),
            'system_ram_peak_above_idle_gib': maximum('ram_used_gib_peak_minus_idle'),
            'gpu_util_peak_pct': maximum('gpu_util_pct_peak'),
            'gpu_util_mean_of_request_means_pct': average('gpu_util_pct_mean'),
            'gpu_vram_peak_mib': maximum('gpu_memory_used_mib_peak'),
            'gpu_vram_peak_above_idle_mib': maximum('gpu_memory_used_mib_peak_minus_idle'),
            'gpu_power_peak_w_unverified': maximum('gpu_power_w_peak'),
            'truncated_requests': sum(r.get('ollama_done_reason') == 'length' for r in records),
            'empty_visible_responses': sum(not (r.get('content') or '').strip() for r in records),
            'disk_free_gib_at_end': shutil.disk_usage(root).free / 1024**3,
            'resource_warning': 'System-wide measurements; other processes may contribute. Prior Ollama models are unloaded between model arms; do not interpret as model-exclusive allocation.',
        }
        save_json(output / f'{name}_metrics.json', {
            'model': name, 'model_tag': tag, 'model_fingerprint': fingerprint,
            'reference_sha256': dataset_hash, 'metrics': overall,
            'per_field': by_field, 'per_document_counts': by_doc,
            'resources': resource_summary,
        })
        summary[name] = {'model': tag, 'canonical_micro_f1': overall['micro_f1'],
                         'precision': overall['precision'], 'recall': overall['recall'],
                         'n': overall['n'], 'gold_present': overall['gold_present'],
                         'schema_key_failure_groups': overall['schema_key_failure_groups'],
                         **resource_summary}
        save_json(output / 'comparison_summary.json', summary)
        csv_write(output / 'comparison_summary.csv', list(summary.values()))
        csv_write(output / f'{name}_requests.csv', [
            {'document_id': r['document_id'], 'field_group': r['field_group'],
             'wall_seconds': r['request_wall_seconds'],
             'prompt_tokens': r.get('ollama_prompt_eval_count'),
             'generated_tokens': r.get('ollama_eval_count'),
             'cpu_peak_pct': r['resource'].get('cpu_util_pct_peak'),
             'ram_peak_gib': r['resource'].get('ram_used_gib_peak'),
             'gpu_peak_pct': r['resource'].get('gpu_util_pct_peak'),
             'vram_peak_mib': r['resource'].get('gpu_memory_used_mib_peak')}
            for r in records])
        # Unload completed model before next model; avoid cumulative VRAM occupancy.
        ollama_post(api + '/api/generate', {'model': tag, 'prompt': '', 'stream': False, 'raw': True, 'keep_alive': 0}, timeout=120)
        print(f'COMPLETED {name}: micro-F1={overall["micro_f1"]:.4f}, '
              f'mean={mean(durations):.1f}s, VRAMpeak={resource_summary["gpu_vram_peak_mib"]}', flush=True)
    print('RESULTS:', output, flush=True)


if __name__ == '__main__':
    main()
