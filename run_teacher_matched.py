"""Read-only matched Ollama teacher evaluation for OTSE development silver references.
Run from the OTSE repository root. Does not modify student predictions or datasets.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from otse_auto.core import digest_obj, properties, read_json, read_jsonl, write_json
from otse_auto.evaluation import model_score, bootstrap_delta


def api(url, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Ollama HTTP {exc.code}: {exc.read().decode("utf-8", errors="replace")[:1200]}') from exc


def strict_parse(content):
    try:
        obj = json.loads(content.strip(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='training_outputs/otse-auto-source-all-v1')
    ap.add_argument('--model', default='qwen3:14b-otse8k')
    ap.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    ap.add_argument('--limit', type=int, default=0, help='Only run first N tasks for smoke testing (0=all).')
    ap.add_argument('--score-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.output).resolve()
    dev = out / 'dataset' / 'dev.jsonl'
    rows = read_jsonl(dev)
    if not rows:
        ap.error(f'No development rows found: {dev}')
    identities = {(r['document_id'], r['field_group']) for r in rows}
    if len(identities) != len(rows):
        raise RuntimeError('Duplicate document/field-group task identifiers')
    pred_dir = out / 'predictions' / 'teacher_14b_matched'
    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir = out / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        base_url = args.ollama_url.rstrip('/')
        tags = api(base_url + '/api/tags', timeout=15)
        models = {m['name']: m for m in tags.get('models', [])}
        if args.model not in models:
            raise RuntimeError(f'Model {args.model!r} missing from Ollama. Available: {list(models)}')
        version = api(base_url + '/api/version', timeout=15).get('version', 'unknown')
        settings = {
            'runtime': 'Ollama_Q4_K_M', 'model': args.model,
            'model_digest': models[args.model].get('digest', ''),
            'ollama_version': version, 'temperature': 0, 'seed': 42,
            'max_tokens': 1024, 'context': 8192,
            'constrained_json': True, 'think': False,
            'top_k': 20, 'top_p': 0.95,
            'evidence': 'exact_dev_messages',
        }
        selected = rows[:args.limit] if args.limit else rows
        for i, row in enumerate(selected, 1):
            path = pred_dir / f"{row['document_id']}__{row['field_group']}.json"
            key = digest_obj([settings, row['source_view_sha256']])
            if path.exists():
                old = read_json(path)
                if old.get('prediction_key') == key and 'parsed' in old:
                    print(f'teacher: {i}/{len(selected)} cached', flush=True)
                    continue
            defs = properties(row['field_group'])
            schema = {'type': 'object', 'properties': defs, 'required': list(defs), 'additionalProperties': False}
            payload = {
                'model': args.model, 'messages': row['messages'][:-1],
                'stream': False, 'think': False, 'format': schema,
                'keep_alive': '15m',
                'options': {'temperature': 0, 'seed': 42, 'num_ctx': 8192,
                            'num_predict': 1024, 'top_k': 20, 'top_p': 0.95},
            }
            started = time.perf_counter()
            response = api(base_url + '/api/chat', payload, timeout=900)
            elapsed = time.perf_counter() - started
            content = response.get('message', {}).get('content') or ''
            parsed = strict_parse(content)
            result = {
                'document_id': row['document_id'], 'field_group': row['field_group'],
                'split': 'dev', 'prediction_key': key, 'settings': settings,
                'source_view_sha256': row['source_view_sha256'],
                'content': content, 'parsed': parsed,
                'schema_keys_ok': isinstance(parsed, dict) and set(parsed) == set(defs),
                'possible_length_cutoff': response.get('done_reason') == 'length',
                'request_wall_seconds': elapsed,
                'usage': {'prompt_tokens': response.get('prompt_eval_count'),
                          'completion_tokens': response.get('eval_count'),
                          'prompt_eval_duration_ns': response.get('prompt_eval_duration'),
                          'eval_duration_ns': response.get('eval_duration')},
            }
            write_json(path, result)
            print(f'teacher: {i}/{len(selected)} dev groups; {elapsed:.2f}s; schema={result["schema_keys_ok"]}', flush=True)

    missing = [r for r in rows if not (pred_dir / f"{r['document_id']}__{r['field_group']}.json").exists()]
    if missing:
        print(f'{len(missing)} of {len(rows)} tasks have no teacher prediction. Results saved; rerun without --limit to finish.', flush=True)
        return 0
    # Verify all predictions correspond to the current model/version before scoring.
    if not args.score_only:
        for row in rows:
            path = pred_dir / f"{row['document_id']}__{row['field_group']}.json"
            pred = read_json(path)
            if pred.get('prediction_key') != digest_obj([settings, row['source_view_sha256']]):
                raise RuntimeError(f'Stale teacher prediction: {path}. Re-run full benchmark to refresh.')
    teacher, by_field, by_doc, _, field_rows = model_score(rows, pred_dir)
    student_dir = out / 'predictions' / 'gguf_baseline'
    student, _, student_by_doc, _, _ = model_score(rows, student_dir)
    result = {
        'scope': 'AUTOMATED_REFERENCE_DEV_PROXY_ONLY_NOT_FULL60',
        'human_verified': False, 'independent_test_claim': False,
        'documents_evaluated': len({r['document_id'] for r in rows}),
        'field_group_tasks': len(rows),
        'known_field_decisions': sum(len(r['known_fields']) for r in rows),
        'teacher': teacher, 'student_gguf': student,
        'teacher_by_field': by_field,
        'paired_document_bootstrap_delta_student_minus_teacher': bootstrap_delta(by_doc, student_by_doc),
        'warning': 'Machine-generated silver references. Runtime differs (Ollama vs llama.cpp); latency excludes model loading and PDF retrieval.',
    }
    write_json(report_dir / 'teacher_vs_student_matched.json', result)
    with (report_dir / 'teacher_vs_student_matched.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'precision', 'recall', 'micro_f1', 'macro_field_f1', 'schema_key_failure_groups', 'total_request_seconds', 'mean_request_seconds'])
        for name, m in [('teacher_14b', teacher), ('student_4b_gguf', student)]:
            writer.writerow([name, m['precision'], m['recall'], m['micro_f1'], m['macro_field_f1'], m['schema_key_failure_groups'], m['inference_request_seconds_total'], m['inference_request_seconds_total'] / len(rows)])
    print(json.dumps({'teacher': {k: teacher[k] for k in ('precision','recall','micro_f1','schema_key_failure_groups','inference_request_seconds_total')},
                      'student': {k: student[k] for k in ('precision','recall','micro_f1','schema_key_failure_groups','inference_request_seconds_total')},
                      'report': str(report_dir / 'teacher_vs_student_matched.json')}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
