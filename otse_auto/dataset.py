"""Partial full-group SFT with explicit value masks; unknown never means absent."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
from .core import GROUPS, LABEL_PROVENANCE, digest_obj, messages_for, read_json, write_json, write_jsonl
from .evidence import alternate_order, serialize_blocks, quote_present


def accepted_fields(annotation, balance=False):
    fields = annotation['fields']
    positives = [f for f in GROUPS[annotation['field_group']] if fields[f]['state'] == 'supported']
    negatives = [f for f in GROUPS[annotation['field_group']] if fields[f]['state'] == 'absent_in_input']
    if balance:
        negatives.sort(key=lambda f: digest_obj([annotation['document_id'], f]))
        negatives = negatives[:max(3, 2 * len(positives))]
    return positives + negatives


def render_answer(group, fields, known):
    known_set = set(known)
    # Put supervised keys first. Unknown placeholders therefore cannot condition any
    # subsequent supervised value through teacher forcing. JSON key order is immaterial.
    order = [f for f in GROUPS[group] if f in known_set] + [f for f in GROUPS[group] if f not in known_set]
    chunks = ['{\n']
    spans = []
    unknown_spans = []
    length = 2
    for i, field in enumerate(order):
        value = fields[field]['value'] if field in known_set else None
        segment = '  ' + json.dumps(field) + ': ' + json.dumps(value, ensure_ascii=False, allow_nan=False)
        start = length
        chunks.append(segment)
        length += len(segment)
        (spans if field in known_set else unknown_spans).append((start, length))
        ending = ',\n' if i < len(order) - 1 else '\n'
        chunks.append(ending)
        length += len(ending)
    chunks.append('}')
    answer = ''.join(chunks)
    if known_set == set(GROUPS[group]):
        spans = [(0, len(answer))]
    return answer, spans, unknown_spans


def token_record(record, tokenizer, max_length):
    messages = record['messages']
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    answer = messages[-1]['content']
    if not full.startswith(prefix + answer):
        raise ValueError('Chat-template assistant prefix mismatch; refusing unsafe label masking')
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = encoded['input_ids']
    if len(ids) > max_length:
        raise ValueError('Example exceeds max_length; no tokens were truncated')
    base = len(prefix)
    spans = [(base + a, base + b) for a, b in record['supervised_answer_spans']]
    unknown = [(base + a, base + b) for a, b in record['unknown_answer_spans']]
    labels = []
    for token, (start, end) in zip(ids, encoded['offset_mapping']):
        allowed = end > start and any(start >= a and end <= b for a, b in spans)
        if allowed and any(start < b and end > a for a, b in unknown):
            raise AssertionError('Unknown value entered training labels')
        labels.append(token if allowed else -100)
    if not any(v != -100 for v in labels):
        raise ValueError('No supervised tokens after masking')
    # Supervise the end-of-turn marker ONLY on fully resolved groups.
    if record.get('complete_group'):
        suffix_start = base + len(answer)
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        for i, (token, offset) in enumerate(zip(ids, encoded['offset_mapping'])):
            if token == eos_id and (offset[0] >= suffix_start or i == len(ids) - 1):
                labels[i] = token
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def make_record(annotation, tokenizer, max_length, augment=False, balance=True):
    known = accepted_fields(annotation, balance)
    if not known:
        return None
    blocks = alternate_order(annotation['blocks']) if augment else annotation['blocks']
    if augment and blocks == annotation['blocks']:
        return None
    by_id = {b['id']: b for b in blocks}
    for field in known:
        for ev in annotation['fields'][field].get('evidence', []):
            if ev['block_id'] not in by_id or not quote_present(ev['quote'], by_id[ev['block_id']]['text']):
                raise ValueError('Evidence lost during augmentation')
    source = serialize_blocks(blocks)
    answer, spans, unknown = render_answer(annotation['field_group'], annotation['fields'], known)
    record = {k: annotation[k] for k in ('document_id', 'sha256', 'split', 'family_id', 'field_group')}
    record.update({'source_view_sha256': digest_obj(source), 'label_provenance': LABEL_PROVENANCE,
        'human_verified': False, 'reference_scope': 'EXACT_SELECTED_INPUT_ONLY',
        'known_fields': known, 'target_values': {f: annotation['fields'][f]['value'] for f in known},
        'target_states': {f: annotation['fields'][f]['state'] for f in known},
        'messages': messages_for(annotation['field_group'], source) + [{'role': 'assistant', 'content': answer}],
        'supervised_answer_spans': spans, 'unknown_answer_spans': unknown,
        'complete_group': len(known) == len(GROUPS[annotation['field_group']]),
        'augmentation': 'whole_block_order' if augment else 'none'})
    tokenized = token_record(record, tokenizer, max_length)
    record['sequence_tokens'] = len(tokenized['input_ids'])
    record['supervised_tokens'] = sum(x != -100 for x in tokenized['labels'])
    record['example_id'] = digest_obj([record['document_id'], record['field_group'], record['source_view_sha256'], record['target_values']])
    return record


def summarize_coverage(annotations):
    result = {}
    for split in ('train', 'dev'):
        selected = [a for a in annotations if a['split'] == split]
        states = Counter(r['state'] for a in selected for r in a['fields'].values())
        positive_docs = {a['document_id'] for a in selected if any(x['state'] == 'supported' for x in a['fields'].values())}
        result[split] = {'documents': len({a['document_id'] for a in selected}), 'groups': len(selected),
                         'documents_with_positive_labels': len(positive_docs), 'field_states': dict(states)}
    return result


def build(out, tokenizer, max_length=4096):
    out = Path(out)
    annotations = [read_json(p) for p in sorted((out / 'annotations').rglob('*.json'))]
    annotations = [a for a in annotations if a.get('label_provenance') == LABEL_PROVENANCE]
    records = {'train': [], 'dev': []}
    negative_only = []
    rejected = []
    seen = set()
    provenance = []
    for ann in annotations:
        if ann['split'] not in records:
            raise ValueError('Forbidden split in automatic annotations')
        try:
            rec = make_record(ann, tokenizer, max_length, balance=ann['split'] == 'train')
        except ValueError as exc:
            rejected.append({'document_id': ann['document_id'], 'group': ann['field_group'], 'reason': str(exc)})
            continue
        if not rec:
            continue
        if rec['example_id'] in seen:
            continue
        seen.add(rec['example_id'])
        positive = any(ann['fields'][f]['state'] == 'supported' for f in rec['known_fields'])
        if ann['split'] == 'train' and not positive:
            negative_only.append(rec)
        else:
            records[ann['split']].append(rec)
        for f, obj in ann['fields'].items():
            provenance.append({'document_id': ann['document_id'], 'split': ann['split'], 'family_id': ann['family_id'],
                               'sha256': ann['sha256'], 'field_group': ann['field_group'], 'field': f,
                               **obj, 'human_verified': False})
        # At most one augmentation on a deterministic quarter of positive train groups.
        if ann['split'] == 'train' and positive and int(digest_obj([ann['document_id'], ann['field_group']])[:8], 16) % 4 == 0:
            try:
                extra = make_record(ann, tokenizer, max_length, augment=True)
                if extra and extra['example_id'] not in seen:
                    records['train'].append(extra)
                    seen.add(extra['example_id'])
            except ValueError:
                pass
    # Prevent absence-only examples dominating the training mix.
    cap = max(0, len(records['train']) // 9)
    records['train'].extend(sorted(negative_only, key=lambda x: x['example_id'])[:cap])
    t_ids = {r['document_id'] for r in records['train']}
    d_ids = {r['document_id'] for r in records['dev']}
    t_f = {r['family_id'] for r in records['train']}
    d_f = {r['family_id'] for r in records['dev']}
    t_s = {r['sha256'] for r in records['train']}
    d_s = {r['sha256'] for r in records['dev']}
    if t_ids & d_ids or t_f & d_f or t_s & d_s:
        raise ValueError('Train/dev leakage detected during dataset construction')
    report = {'label_provenance': LABEL_PROVENANCE, 'human_verified': False,
              'reference_scope': 'EXACT_SELECTED_INPUT_ONLY', 'max_length': max_length,
              'annotation_coverage': summarize_coverage(annotations), 'splits': {}, 'rejected': rejected}
    for split, rows in records.items():
        count = Counter(f for r in rows if r['augmentation'] == 'none' for f in r['known_fields'])
        values = Counter(f for r in rows if r['augmentation'] == 'none' for f in r['known_fields'] if r['target_states'][f] == 'supported')
        report['splits'][split] = {'examples': len(rows), 'documents': len({r['document_id'] for r in rows}),
           'families': len({r['family_id'] for r in rows}), 'augmented_examples': sum(r['augmentation'] != 'none' for r in rows),
           'positive_documents': len({r['document_id'] for r in rows if any(s == 'supported' for s in r['target_states'].values())}),
           'complete_group_examples': sum(r['complete_group'] for r in rows),
           'known_field_counts': dict(count), 'positive_field_counts': dict(values),
           'positive_groups_by_type': {g: len({r['document_id'] for r in rows if r['field_group'] == g and any(v == 'supported' for v in r['target_states'].values())}) for g in GROUPS},
           'sequence_max_tokens': max((r['sequence_tokens'] for r in rows), default=0)}
    report['ready_for_training'] = (report['splits']['train']['positive_documents'] >= 50
                                    and report['splits']['dev']['positive_documents'] >= 12
                                    and sum(report['splits']['train']['positive_field_counts'].values()) >= 150
                                    and sum(report['splits']['dev']['positive_field_counts'].values()) >= 30
                                    and all(n >= 20 for n in report['splits']['train']['positive_groups_by_type'].values())
                                    and all(n >= 5 for n in report['splits']['dev']['positive_groups_by_type'].values()))
    report['dataset_signature'] = digest_obj({s: [r['example_id'] for r in rows] for s, rows in records.items()})
    for split, rows in records.items():
        write_jsonl(out / 'dataset' / f'{split}.jsonl', rows)
    write_jsonl(out / 'dataset' / 'field_provenance.jsonl', provenance)
    write_json(out / 'dataset' / 'dataset_report.json', report)
    return report
