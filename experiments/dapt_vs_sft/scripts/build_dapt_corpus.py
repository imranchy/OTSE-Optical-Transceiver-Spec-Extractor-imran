#!/usr/bin/env python3
"""Build a leakage-conscious OTSE DAPT pilot corpus from local PDFs.

Run from the OTSE repository root:
 python experiments/dapt_vs_sft/scripts/build_dapt_corpus.py

Requires PyMuPDF (fitz); standard-library otherwise.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import random
import re
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as fh:
        for line_number, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def clean_page(text):
    lines = []
    for line in text.replace('\x00', ' ').splitlines():
        line = re.sub(r'[ \t]+', ' ', line).strip()
        if line:
            lines.append(line)
    return '\n'.join(lines)


def extract_pdf(path, max_pages, min_chars):
    import fitz
    pages = []
    with fitz.open(path) as doc:
        for i in range(min(len(doc), max_pages)):
            text = clean_page(doc[i].get_text('text', sort=True))
            if len(text) >= min_chars:
                pages.append({'page': i + 1, 'text': text})
    return pages


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--max-documents', type=int, default=100)
    parser.add_argument('--validation-documents', type=int, default=10)
    parser.add_argument('--max-pages', type=int, default=30)
    parser.add_argument('--min-page-chars', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strict', action='store_true', help='Fail if any selected PDF cannot be validated/extracted')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / 'otse_final_reference_v1/provenance/corpus_split_manifest.csv'
    pdf_root = root / 'data/corpus_v2/raw_pdfs'
    sft_root = root / 'data/corpus_v2/pseudo_qlora'
    output = root / 'experiments/dapt_vs_sft/data'
    output.mkdir(parents=True, exist_ok=True)
    for p in (manifest, pdf_root, sft_root / 'train.jsonl', sft_root / 'validation.jsonl'):
        if not p.exists():
            raise FileNotFoundError(p)
    if args.max_documents < 2 or not 1 <= args.validation_documents < args.max_documents:
        raise ValueError('Require max-documents >= 2 and 1 <= validation-documents < max-documents')
    with manifest.open(encoding='utf-8-sig', newline='') as fh:
        records = list(csv.DictReader(fh))
    required = {'document_id','relative_path','sha256','family_id','split','technology','form_factor'}
    if not records or not required.issubset(records[0]):
        raise ValueError('Manifest missing required columns')
    by_id = {}
    family_splits = collections.defaultdict(set)
    for r in records:
        if r['document_id'] in by_id:
            raise ValueError('Duplicate document ID in manifest: ' + r['document_id'])
        by_id[r['document_id']] = r
        family_splits[r['family_id']].add(r['split'])
    if any(len(v) > 1 for v in family_splits.values()):
        raise ValueError('Manifest has cross-split family IDs')
    train_ids = {r['document_id'] for r in read_jsonl(sft_root / 'train.jsonl')}
    validation_ids = {r['document_id'] for r in read_jsonl(sft_root / 'validation.jsonl')}
    if train_ids & validation_ids:
        raise ValueError('SFT train/validation document overlap')
    unknown = (train_ids | validation_ids) - set(by_id)
    if unknown:
        raise ValueError(f'{len(unknown)} SFT IDs absent from manifest')
    if any(by_id[d]['split'] != 'train' for d in train_ids | validation_ids):
        raise ValueError('SFT records mapped outside manifest train split')
    # Exclude original SFT validation documents AND all their recorded family members.
    blocked_families = {by_id[d]['family_id'] for d in validation_ids}
    eligible = [r for r in records if r['split'] == 'train' and r['document_id'] not in validation_ids and r['family_id'] not in blocked_families]
    # Deterministic round-robin across technology/form factor strata.
    rng = random.Random(args.seed)
    strata = collections.defaultdict(list)
    for r in eligible:
        strata[(r['technology'], r['form_factor'])].append(r)
    for bucket in strata.values():
        rng.shuffle(bucket)
    keys = sorted(strata)
    rng.shuffle(keys)
    ordered = []
    while any(strata.values()):
        for key in keys:
            if strata[key]:
                ordered.append(strata[key].pop())
    selected = []
    rejected = []
    seen_sha = set()
    for r in ordered:
        if len(selected) >= args.max_documents:
            break
        rel = Path(r['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            rejected.append((r['document_id'], 'unsafe_relative_path'))
            continue
        path = pdf_root / rel
        if not path.is_file():
            rejected.append((r['document_id'], 'missing_pdf'))
            continue
        actual = sha256(path)
        if actual.lower() != r['sha256'].lower():
            rejected.append((r['document_id'], 'sha256_mismatch'))
            continue
        if actual in seen_sha:
            rejected.append((r['document_id'], 'duplicate_sha256'))
            continue
        try:
            pages = extract_pdf(path, args.max_pages, args.min_page_chars)
        except Exception as exc:
            rejected.append((r['document_id'], 'pdf_error:' + str(exc)[:100]))
            continue
        if not pages:
            rejected.append((r['document_id'], 'no_usable_text'))
            continue
        seen_sha.add(actual)
        selected.append((r, pages))
    if args.strict and (rejected or len(selected) < args.max_documents):
        raise RuntimeError(f'Strict mode: {len(rejected)} rejected, {len(selected)} selected')
    if len(selected) < 2:
        raise RuntimeError('Too few valid PDFs to create train and validation corpus')
    # Split at document level. Keep family members together within the DAPT pilot.
    families = collections.defaultdict(list)
    for r, pages in selected:
        families[r['family_id']].append((r, pages))
    fam_keys = sorted(families)
    rng.shuffle(fam_keys)
    valid_fams = set()
    valid_count = 0
    for fam in fam_keys:
        if valid_count >= args.validation_documents:
            break
        if len(selected) - valid_count - len(families[fam]) < 1:
            continue
        valid_fams.add(fam)
        valid_count += len(families[fam])
    train_rows, val_rows, provenance = [], [], []
    for r, pages in selected:
        partition = 'validation' if r['family_id'] in valid_fams else 'train'
        target = train_rows if partition == 'train' else val_rows
        for page in pages:
            target.append({'text': page['text'], 'document_id': r['document_id'], 'page': page['page']})
        provenance.append({**r, 'dapt_partition': partition, 'usable_pages': len(pages)})
    if not train_rows or not val_rows:
        raise RuntimeError('Empty train or validation dataset')
    write_jsonl(output / 'dapt_train.jsonl', train_rows)
    write_jsonl(output / 'dapt_validation.jsonl', val_rows)
    with (output / 'corpus_provenance.csv').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)
    audit = {
        'seed': args.seed,
        'manifest_documents': len(records),
        'manifest_splits': dict(collections.Counter(r['split'] for r in records)),
        'original_sft_train_documents': len(train_ids),
        'original_sft_validation_documents_excluded': len(validation_ids),
        'eligible_documents': len(eligible),
        'selected_documents': len(selected),
        'dapt_train_documents': sum(r['dapt_partition'] == 'train' for r in provenance),
        'dapt_validation_documents': sum(r['dapt_partition'] == 'validation' for r in provenance),
        'dapt_train_pages': len(train_rows),
        'dapt_validation_pages': len(val_rows),
        'rejections': [{'document_id': d, 'reason': why} for d, why in rejected],
        'notes': [
            'Document IDs and recorded family IDs are separated; semantic near-duplicates still require inspection.',
            'Only train-manifest PDFs used. Original SFT validation, dev, test, reserve excluded.',
            'DAPT text is PDF-extracted, not independently verified for table-reading quality.',
            'No held-out extraction labels are used in DAPT corpus creation.'
        ]
    }
    (output / 'leakage_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k != 'rejections'}, indent=2))
    print(f'Rejected candidates: {len(rejected)}')
    print('Saved to:', output)

if __name__ == '__main__':
    main()
