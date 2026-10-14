"""Match expert-reviewed OTSE report runs to corpus PDFs without changing gold labels.
Run from the original OTSE repository root.
Requires: PyMuPDF (fitz). Standard-library dependencies otherwise.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import fitz

ROOT = Path.cwd()
TOKEN = re.compile(r"[a-z0-9]+(?:[.+/-][a-z0-9]+)*", re.I)


def tokens(s):
    return TOKEN.findall(s.lower())


def signature(s):
    """A set of distinctive 5-token phrases, excluding short/noisy tokens."""
    t = [x for x in tokens(s) if len(x) >= 3]
    return set(' '.join(t[i:i+5]) for i in range(max(0, len(t)-4)))


def read_json(p):
    try:
        return json.loads(p.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def text_pdf(p):
    with fitz.open(p) as doc:
        return '\n'.join(page.get_text('text', sort=True) for page in doc)


def dump_json(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(p, rows, fields):
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='otse_final_reference_v1/provenance/corpus_split_manifest.csv')
    ap.add_argument('--pdf-root', default='data/corpus_v2/raw_pdfs')
    ap.add_argument('--reports', default='data/expert_gold/reports')
    ap.add_argument('--output', default='experiments/dapt_vs_sft/results/gold_benchmark')
    ap.add_argument('--min-overlap', type=int, default=12, help='Minimum shared distinctive phrases')
    ap.add_argument('--min-coverage', type=float, default=0.12, help='Minimum report phrase coverage')
    ap.add_argument('--min-margin', type=float, default=1.35, help='Best/runner-up score ratio')
    args = ap.parse_args()
    manifest = ROOT / args.manifest
    pdfroot = ROOT / args.pdf_root
    reports = ROOT / args.reports
    out = ROOT / args.output
    if not manifest.is_file() or not pdfroot.is_dir() or not reports.is_dir():
        raise SystemExit('Missing manifest, PDF directory, or reports directory; check paths.')
    out.mkdir(parents=True, exist_ok=True)
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        entries = list(csv.DictReader(f))
    required = {'document_id','relative_path','sha256','family_id','split'}
    if not entries or not required.issubset(entries[0]):
        raise SystemExit('Manifest missing required columns: '+str(required))

    # Hash validation and text index. No PDF content or gold labels are modified.
    valid = {}
    hash_to_ids = collections.defaultdict(list)
    filename_to_ids = collections.defaultdict(list)
    inverted = collections.defaultdict(set)
    sigs = {}
    pdf_issues = []
    for i, row in enumerate(entries, 1):
        did = row['document_id']
        rel = Path(row['relative_path'].replace('\\','/'))
        if rel.is_absolute() or '..' in rel.parts:
            pdf_issues.append({'document_id':did,'issue':'unsafe_relative_path'}); continue
        p = pdfroot / rel
        if not p.is_file():
            pdf_issues.append({'document_id':did,'issue':'missing_pdf','path':str(p)}); continue
        try:
            sha = digest(p)
            if sha.lower() != row['sha256'].lower():
                pdf_issues.append({'document_id':did,'issue':'hash_mismatch','path':str(p)}); continue
            sg = signature(text_pdf(p))
        except Exception as exc:
            pdf_issues.append({'document_id':did,'issue':'pdf_read_error','detail':str(exc)}); continue
        if not sg:
            pdf_issues.append({'document_id':did,'issue':'empty_pdf_text'}); continue
        valid[did] = (row, p)
        hash_to_ids[sha.lower()].append(did)
        filename_to_ids[p.name.lower()].append(did)
        sigs[did] = sg
        for gram in sg:
            inverted[gram].add(did)
        if i % 100 == 0:
            print('Indexed',i,'/',len(entries),'manifest rows',flush=True)

    runs = sorted(reports.rglob('specs.json'))
    rows = []
    for j, specpath in enumerate(runs, 1):
        run = specpath.parent
        meta = read_json(run/'meta.json') or {}
        parsed = read_json(run/'parsed.json') or {}
        spec = read_json(specpath)
        source = str(meta.get('source_file') or parsed.get('source_path') or '')
        source_basename = re.split(r'[\\/]', source)[-1].lower()
        status = 'unmatched'
        reason = ''
        method = ''
        candidates = []
        coverage = 0.0
        margin = 0.0
        matched = ''
        if not isinstance(spec, dict):
            reason = 'invalid_specs_json'
        else:
            # A real PDF path can be used for exact SHA matching, if accessible.
            possible = Path(source)
            if possible.is_file():
                try:
                    candidates = [(d,1e9) for d in hash_to_ids.get(digest(possible), [])]
                    if candidates: method = 'sha256'
                except OSError:
                    pass
            # Filename-only matches are not accepted without corroborating content.
            content_path = run/'content.md'
            content = content_path.read_text(encoding='utf-8',errors='replace') if content_path.exists() else ''
            rs = signature(content)
            if rs and not candidates:
                counter = collections.Counter()
                for gram in rs:
                    counter.update(inverted.get(gram, ()))
                candidates = counter.most_common(3)
                method = 'text_5gram'
                if candidates:
                    coverage = candidates[0][1]/len(rs)
                    margin = candidates[0][1]/max(1,candidates[1][1]) if len(candidates)>1 else float('inf')
            if len(candidates)==1 and method=='sha256':
                matched = candidates[0][0]; status='matched';reason='verified_hash'
            elif method=='text_5gram' and candidates:
                top, count = candidates[0]
                # Require report coverage, absolute evidence and separation from runner-up.
                if count >= args.min_overlap and coverage >= args.min_coverage and margin >= args.min_margin:
                    matched=top;status='matched';reason='strong_text_match'
                else:
                    status='ambiguous';reason='weak_or_competing_text_match'
            elif method=='sha256' and len(candidates)>1:
                status='ambiguous';reason='duplicate_sha_in_manifest'
            elif not content:
                reason='missing_report_content'
            else:
                reason='no_matching_pdf'
            if matched and source_basename and source_basename in filename_to_ids:
                if matched not in filename_to_ids[source_basename]:
                    # Some colleague paths are temporary, so only flag, do not reject.
                    reason += ';source_filename_differs'
        row, p = valid.get(matched, ({},None))
        rows.append({
            'run_id':run.name,'specs_json':str(specpath.relative_to(ROOT)),
            'source_file':source,'status':status,'reason':reason,'match_method':method,
            'document_id':matched,'split':row.get('split',''),
            'family_id':row.get('family_id',''),'pdf_relative_path':row.get('relative_path',''),
            'pdf_sha256':row.get('sha256',''),
            'shared_phrases':candidates[0][1] if candidates and method=='text_5gram' else '',
            'report_phrase_coverage':round(coverage,4),
            'top_vs_second_ratio':round(margin,3) if margin!=float('inf') else 'inf',
            'second_candidate_id':candidates[1][0] if len(candidates)>1 else '',
            'second_candidate_shared_phrases':candidates[1][1] if len(candidates)>1 else '',
            'gold_top_level_fields':len(spec) if isinstance(spec,dict) else 0,
            'validation_passed':meta.get('validation_passed',''),
        })
        if j % 50 == 0:
            print('Matched reports',j,'/',len(runs),flush=True)

    # Repeated runs of a single PDF are not independent test samples. If gold
    # JSON differs across runs, exclude until a reviewer selects the final version.
    by_doc = collections.defaultdict(list)
    for r in rows:
        if r['status']=='matched':
            by_doc[r['document_id']].append(r)
    accepted=[]
    duplicates=[]
    conflicts=[]
    for did, rr in by_doc.items():
        rr.sort(key=lambda x:x['run_id'])
        payloads = [read_json(ROOT/x['specs_json']) for x in rr]
        canonical = [json.dumps(v,sort_keys=True,ensure_ascii=False) for v in payloads]
        if len(set(canonical))>1:
            for r in rr:
                r['status']='conflicting_gold_runs'
                r['reason']='same_pdf_different_specs_json'
            conflicts.append({'document_id':did,'run_ids':[r['run_id'] for r in rr]})
            continue
        chosen=rr[-1]
        accepted.append(chosen)
        for r in rr[:-1]:
            r['status']='duplicate_run'
            r['reason']='same_pdf_same_gold_json'
            duplicates.append(r['run_id'])

    # Check recorded family IDs do not cross manifest partitions.
    fam = collections.defaultdict(set)
    for e in entries:
        fam[e['family_id']].add(e['split'])
    family_conflicts = [f for f,s in fam.items() if len(s)>1]
    test_rows = [r for r in accepted if r['split']=='test' and r['family_id'] not in family_conflicts]
    test_rows.sort(key=lambda r:r['document_id'])
    output_fields = ['run_id','specs_json','source_file','status','reason','match_method',
                     'document_id','split','family_id','pdf_relative_path','pdf_sha256',
                     'shared_phrases','report_phrase_coverage','top_vs_second_ratio',
                     'second_candidate_id','second_candidate_shared_phrases',
                     'gold_top_level_fields','validation_passed']
    write_csv(out/'gold_match_manifest.csv',rows,output_fields)
    with (out/'gold_test_manifest.jsonl').open('w',encoding='utf-8') as f:
        for r in test_rows:
            # References only; do not silently transform expert-reviewed values.
            obj={'document_id':r['document_id'],'family_id':r['family_id'],
                 'pdf_path':str(pdfroot/Path(r['pdf_relative_path'])),
                 'pdf_sha256':r['pdf_sha256'],
                 'gold_specs_path':str(ROOT/r['specs_json']),
                 'gold_run_id':r['run_id'],'match_method':r['match_method'],
                 'gold_status':'expert_reviewed_user_confirmed'}
            f.write(json.dumps(obj,ensure_ascii=False)+'\n')
    audit={
        'report_specs_files':len(runs),'manifest_documents':len(entries),
        'indexed_hash_verified_pdfs':len(valid),
        'pdf_issues':pdf_issues,
        'match_status_counts':dict(collections.Counter(r['status'] for r in rows)),
        'unique_nonconflicting_matched_documents':len(accepted),
        'unique_matched_splits':dict(collections.Counter(r['split'] for r in accepted)),
        'eligible_heldout_test_documents':len(test_rows),
        'duplicate_runs_dropped':len(duplicates),
        'conflicting_gold_runs':conflicts,
        'manifest_cross_split_families':family_conflicts,
        'notes':[
            'Gold JSON files were not modified or reinterpreted.',
            'Text matching is a heuristic: inspect gold_match_manifest.csv before paper reporting.',
            'Gold references were described as expert-reviewed by the dataset provider.',
            'An expert-reviewed label does not itself prove no historical benchmark exposure.',
            'No model inference or F1 calculation was performed.'
        ]}
    dump_json(out/'gold_matching_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('pdf_issues','conflicting_gold_runs','manifest_cross_split_families')},indent=2))
    print('Outputs:',out)

if __name__=='__main__':
    main()
