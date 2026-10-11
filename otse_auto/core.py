"""Paths, provenance, immutable run configurations, and project contracts."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
BASE_ADAPTER = 'training_outputs/qwen3-4b-otse-expanded-745/adapter_final'
BASE_GGUF = 'deployment/otse-qwen4b-Q4_K_M.gguf'
DEFAULT_OUT = 'training_outputs/otse-auto-source-v1'
GROUPS = json.loads((HERE / 'groups.json').read_text(encoding='utf-8'))
PROMPTS = json.loads((HERE / 'prompt_snapshot.json').read_text(encoding='utf-8'))
LABEL_PROVENANCE = 'AUTOMATED_SOURCE_CONSENSUS_NOT_HUMAN_GOLD'


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path: Path, value: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def read_jsonl(path: Path):
    if not Path(path).is_file():
        return []
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def write_jsonl(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def digest_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for b in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def inside(root: Path, relative: str):
    path = (root / relative).resolve()
    if path == root.resolve() or root.resolve() not in path.parents:
        raise ValueError('Unsafe or empty relative path: ' + relative)
    return path


def project_root(value=None):
    path = Path(value).resolve() if value else HERE.parent
    required = path / 'data/corpus_v2/manifests/v2_split_metadata_balanced_proposed.csv'
    if not required.is_file():
        raise FileNotFoundError('Copy the otse_auto folder AND CMD files into the existing OTSE repository root. Missing: ' + str(required))
    return path


def load_manifest(root: Path):
    path = root / 'data/corpus_v2/manifests/v2_split_metadata_balanced_proposed.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    required = {'document_id', 'relative_path', 'sha256', 'family_id', 'split'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Manifest missing required columns')
    if len({r['document_id'] for r in rows}) != len(rows) or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Duplicate document ID or source hash in manifest')
    counts = {s: sum(r['split'] == s for r in rows) for s in ('train', 'dev', 'test', 'reserve')}
    if counts != {'train': 450, 'dev': 60, 'test': 60, 'reserve': 46}:
        raise ValueError('Unexpected source split counts: ' + str(counts))
    if any(not r['family_id'].strip() for r in rows):
        raise ValueError('Empty family identifier')
    split_families = {s: {r['family_id'] for r in rows if r['split'] == s} for s in counts}
    names = list(counts)
    for i, s in enumerate(names):
        for t in names[i + 1:]:
            if split_families[s] & split_families[t]:
                raise ValueError('Recorded family overlap between ' + s + ' and ' + t)
    return rows


def eligible_rows(root: Path):
    rows = load_manifest(root)
    old_val = root / 'data/corpus_v2/pseudo_qlora/validation.jsonl'
    if not old_val.is_file():
        raise FileNotFoundError('Legacy internal-validation exclusion file is required: ' + str(old_val))
    val_ids = {r['document_id'] for r in read_jsonl(old_val)}
    if not val_ids.issubset({r['document_id'] for r in rows if r['split'] == 'train'}):
        raise ValueError('Legacy validation IDs not entirely in the train partition')
    excluded_families = {r['family_id'] for r in rows if r['document_id'] in val_ids}
    train = [r for r in rows if r['split'] == 'train' and r['family_id'] not in excluded_families]
    dev = [r for r in rows if r['split'] == 'dev']
    return train, dev, {'legacy_internal_validation_ids': sorted(val_ids), 'excluded_train_families': sorted(excluded_families),
                        'eligible_train_documents': len(train), 'eligible_dev_documents': len(dev),
                        'family_checks': 'Recorded manifest families only; not a semantic proof of independence.'}


def diverse_order(rows):
    # Stable round-robin technology/form-factor buckets, one family before repeats.
    buckets = {}
    for r in sorted(rows, key=lambda x: digest_obj(x['document_id'])):
        key = (r.get('technology', ''), r.get('form_factor', ''))
        buckets.setdefault(key, []).append(r)
    first, repeats, seen = [], [], set()
    while any(buckets.values()):
        for key in sorted(buckets):
            if not buckets[key]:
                continue
            r = buckets[key].pop(0)
            if r['family_id'] in seen:
                repeats.append(r)
            else:
                first.append(r)
                seen.add(r['family_id'])
    return first + repeats


def source_path(root: Path, row):
    if row['split'] not in ('train', 'dev'):
        raise ValueError('Automated labeler cannot access test/reserve sources')
    path = inside(root / 'data/corpus_v2/raw_pdfs', row['relative_path'])
    if not path.is_file():
        raise FileNotFoundError('Source PDF missing: ' + str(path))
    if sha256(path) != row['sha256']:
        raise ValueError('Source PDF SHA256 mismatch: ' + str(path))
    return path


def model_class():
    from .schema_snapshot import TransceiverSpecs
    return TransceiverSpecs


def check_project_schema(root):
    from pydantic import TypeAdapter
    path = root / 'transceiver_models.py'
    spec = importlib.util.spec_from_file_location('_otse_live_schema', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    live = module.TransceiverSpecs
    saved = model_class()
    for group, fields in GROUPS.items():
        for field in fields:
            if field not in live.model_fields:
                raise ValueError('Live schema missing field ' + field)
            a = TypeAdapter(live.model_fields[field].annotation).json_schema()
            b = TypeAdapter(saved.model_fields[field].annotation).json_schema()
            if a != b:
                raise ValueError('Live field type changed: ' + field)
    return {'fields': sum(map(len, GROUPS.values())), 'groups': len(GROUPS), 'type_match': True}


def properties(group):
    from pydantic import TypeAdapter
    model = model_class()
    return {f: {**TypeAdapter(model.model_fields[f].annotation).json_schema(),
                'description': model.model_fields[f].description or ''} for f in GROUPS[group]}


def validate_value(field, value):
    from pydantic import TypeAdapter
    if isinstance(value, bool) or (isinstance(value, list) and any(isinstance(v, bool) for v in value)):
        raise ValueError('Booleans are not optical values')
    if isinstance(value, list) and (len(value) > 40 or any(v is None for v in value)):
        raise ValueError('Invalid or overly long value list')
    out = TypeAdapter(model_class().model_fields[field].annotation).validate_python(value, strict=True)
    for v in out if isinstance(out, list) else [out]:
        if isinstance(v, (float, int)) and not math.isfinite(v):
            raise ValueError('Nonfinite value')
    return out


def messages_for(group, source):
    headings = '\n'.join(s for s in source.splitlines() if s.startswith('### PAGE '))
    user = '## Requested fields\n' + PROMPTS['schemas'][group] + '\n\n## Section headings\n' + headings + '\n\n## Datasheet content\n' + source + '\n\nReturn one JSON object containing only the requested fields.\n'
    return [{'role': 'system', 'content': PROMPTS['system']}, {'role': 'user', 'content': user}]


def baseline_fingerprint(root):
    adapter = root / BASE_ADAPTER
    if not (adapter / 'adapter_config.json').is_file():
        raise FileNotFoundError('Preserved original adapter required: ' + str(adapter))
    cfg = read_json(adapter / 'adapter_config.json')
    if cfg.get('base_model_name_or_path', BASE_MODEL).rstrip('/') != BASE_MODEL:
        raise ValueError('Original adapter base model does not match ' + BASE_MODEL)
    files = {str(p.relative_to(root)).replace('\\', '/'): sha256(p) for p in sorted(adapter.rglob('*')) if p.is_file()}
    gguf = root / BASE_GGUF
    if gguf.is_file():
        files[BASE_GGUF] = sha256(gguf)
    return files


def verify_baseline(root, saved):
    for rel, expected in saved.items():
        path = inside(root, rel)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError('Baseline changed outside this workflow: ' + rel)
