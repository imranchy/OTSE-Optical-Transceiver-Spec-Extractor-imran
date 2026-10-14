"""OTSE source-grounded review queue and strict verified-only factual scoring.
Usage: python verify_and_score.py prepare --root . --split test --out paper_results/factual_test
       python verify_and_score.py score --reviews paper_results/factual_test/gold_review.csv --out paper_results/factual_test
Dependencies: PyMuPDF (pip install pymupdf) for PDF text; stdlib otherwise.
"""
import argparse,csv,json,re,math,collections,statistics,hashlib
from pathlib import Path

MODELS={'teacher':'data/corpus_v2/benchmark_v1/{split}/teacher',
        'student':'data/corpus_v2/benchmark_v1/{split}/student',
        'gguf':'data/corpus_v2/benchmark_gguf_constrained_v2/{split}/gguf'}
FIELDS_ORDER=[]
MISSING=object()

def json_str(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def canon(x):
    if x is None:return None
    if isinstance(x,bool):return ('bool',x)
    if isinstance(x,(int,float)):
        if not math.isfinite(x):return ('nonfinite',str(x))
        return ('number',round(float(x),6))
    return ('text',re.sub(r'\s+',' ',str(x).strip()).casefold())
def atom_list(v):
    if v is MISSING:return []
    if v is None:return []
    if isinstance(v,list):return [canon(x) for x in v if x is not None]
    return [canon(v)]
def presence(v):return bool(atom_list(v))
def norm_json(v):
    if v is MISSING:return '<MISSING KEY>'
    return json_str(v)
def load_preds(root,split):
    result={}
    for model,templ in MODELS.items():
        path=root/templ.format(split=split)
        if not path.is_dir():raise FileNotFoundError(f'Missing prediction directory: {path}')
        result[model]={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in path.glob('*.json')}
    ids=set.intersection(*(set(d) for d in result.values()))
    if len(ids)!=60:raise ValueError(f'Expected 60 paired documents; got {len(ids)}; counts: '+str({k:len(v) for k,v in result.items()}))
    return result,sorted(ids)
def flatten(pred):
    fields={};status={}
    for group,rec in pred.get('groups',{}).items():
        d=rec.get('parsed_original')
        if not isinstance(d,dict):d=rec.get('parsed')
        if isinstance(d,dict):
            for k,v in d.items():fields[k]=v;status[k]=bool(rec.get('raw_schema_valid'))
    return fields,status

def load_inventory(root):
    path=root/'data/corpus_v2/manifests/corpus_inventory.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    return {r['document_id']:r for r in rows}
def pages_of(pdf):
    try:
        import fitz
    except ImportError:
        raise SystemExit('Install PDF text extraction: python -m pip install pymupdf')
    doc=fitz.open(str(pdf));pages=[p.get_text('text') for p in doc];doc.close();return pages

def candidate_search(pages,vals):
    """Candidate only: literal mention is NOT verification of field/value relation."""
    terms=[]
    for val in vals:
        if val is MISSING or val is None:continue
        for item in val if isinstance(val,list) else [val]:
            if item is None or isinstance(item,bool):continue
            t=str(item).strip()
            if len(t)>=3 and len(t)<=80 and t.casefold() not in {'none','null','unknown','n/a'}:terms.append(t)
    # limit costly scans; evidence retrieval is advisory, not gold
    for term in terms[:12]:
        for pn,page in enumerate(pages,1):
            match=re.search(re.escape(term),page,re.IGNORECASE)
            if match:
                left=max(0,match.start()-120);right=min(len(page),match.end()+160)
                return pn,' '.join(page[left:right].split())[:350],term
    return '','',''

def prepare(args):
    root=Path(args.root).resolve();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    preds,ids=load_preds(root,args.split);inventory=load_inventory(root)
    fieldnames=['document_id','source_pdf','field','group','teacher_json','student_json','gguf_json',
                'teacher_raw_group_valid','student_raw_group_valid','gguf_raw_group_valid',
                'disagreement','priority','candidate_page','candidate_quote','candidate_search_term',
                'review_status','gold_json','evidence_quote','source_page','reviewer','review_notes']
    rows=[];missing_pdfs=[]
    for idx,doc_id in enumerate(ids,1):
        info=inventory.get(doc_id)
        if not info:raise ValueError('Document not in inventory: '+doc_id)
        pdf=root/'data/corpus_v2/raw_pdfs'/info['relative_path']
        if not pdf.exists():missing_pdfs.append(str(pdf));pages=[]
        else:pages=pages_of(pdf)
        flat={};groups={};status={}
        for model in MODELS:
            pred=preds[model][doc_id]
            flat[model],status[model]=flatten(pred)
            groups[model]={field:g for g,r in pred['groups'].items() for field in ((r.get('parsed_original') if isinstance(r.get('parsed_original'),dict) else r.get('parsed')) or {}).keys()}
        # Include every field even when a model omits the key
        keys=set.union(*(set(flat[m]) for m in MODELS))
        # Read keys from all three groups' prompt outputs and schema when present
        for key in sorted(keys):
            values=[flat[m].get(key,MISSING) for m in MODELS]
            unique={norm_json(v) for v in values};disagree=len(unique)>1
            group=next((groups[m].get(key) for m in MODELS if groups[m].get(key)), '')
            raw_flags=[status[m].get(key,False) for m in MODELS]
            priority='high' if disagree or not all(raw_flags) else 'routine'
            pn,quote,term=candidate_search(pages,values)
            rows.append({'document_id':doc_id,'source_pdf':str(pdf.relative_to(root)).replace('\\','/'),'field':key,'group':group,
                         **{f'{m}_json':norm_json(flat[m].get(key,MISSING)) for m in MODELS},
                         **{f'{m}_raw_group_valid':str(status[m].get(key,False)).lower() for m in MODELS},
                         'disagreement':str(disagree).lower(),'priority':priority,
                         'candidate_page':pn,'candidate_quote':quote,'candidate_search_term':term,
                         'review_status':'pending','gold_json':'','evidence_quote':'','source_page':'','reviewer':'','review_notes':''})
        print(f'Prepared {idx}/{len(ids)}: {doc_id} ({len(keys)} fields)',flush=True)
    # Check field completeness: 48 expected from existing benchmark
    counts=collections.Counter(r['document_id'] for r in rows)
    if any(n!=48 for n in counts.values()):
        raise ValueError('Expected 48 fields per document, got '+str({k:v for k,v in counts.items() if v!=48}))
    with (out/'gold_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
    high=[r for r in rows if r['priority']=='high']
    with (out/'priority_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(high)
    summary={'split':args.split,'documents':len(ids),'rows':len(rows),'high_priority':len(high),'routine':len(rows)-len(high),'missing_pdfs':missing_pdfs,'gold_verified':0,
             'warning':'Candidate page/quote is only a literal search hit, not independently verified ground truth. Do not mark verified without reviewing the PDF and field association.'}
    (out/'preparation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

def load_review(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def parse_gold(row):
    if row['review_status'].strip().lower()!='verified':return MISSING
    if not row['reviewer'].strip() or not row['evidence_quote'].strip() or not row['source_page'].strip():
        raise ValueError(f"Verified row missing reviewer/evidence/page: {row['document_id']} {row['field']}")
    if row['gold_json'].strip()=='':raise ValueError('Verified row has empty gold_json; use null for reviewed absence')
    try:return json.loads(row['gold_json'])
    except json.JSONDecodeError as exc:raise ValueError(f"Bad gold_json {row['document_id']} {row['field']}: {exc}")
def parse_prediction(row,m):
    v=row[f'{m}_json']
    if v=='<MISSING KEY>':return MISSING
    return json.loads(v)
def score(args):
    rows=load_review(Path(args.reviews)); verified=[]
    for row in rows:
        gold=parse_gold(row)
        if gold is not MISSING:verified.append((row,gold))
    if not verified:raise SystemExit('No verified gold rows yet. Review PDFs and fill review_status, gold_json, evidence_quote, source_page and reviewer.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    counts=collections.Counter(r['document_id'] for r,g in verified)
    if len(counts)<60:print(f'WARNING: only {len(counts)}/60 documents have verified rows; metrics are partial and cannot be labeled full-test accuracy')
    details=[];aggregate={}
    for model in MODELS:
        tp=fp=fn=tn=0;exact=0;sum_presence=collections.Counter();byfield=collections.defaultdict(lambda:collections.Counter());unsupported=0;omitted=0
        for row,gold in verified:
            pred=parse_prediction(row,model);gs=collections.Counter(atom_list(gold));ps=collections.Counter(atom_list(pred))
            # Multiset TP/FP/FN: duplicates are not silently ignored
            hit=sum((gs&ps).values());extra=sum((ps-gs).values());miss=sum((gs-ps).values())
            tp+=hit;fp+=extra;fn+=miss;unsupported+=extra;omitted+=miss
            gp=presence(gold);pp=presence(pred)
            if gp and pp:sum_presence['tp']+=1
            elif gp and not pp:sum_presence['fn']+=1
            elif not gp and pp:sum_presence['fp']+=1
            else:sum_presence['tn']+=1
            is_exact=(gs==ps and pred is not MISSING)
            exact+=is_exact
            f=byfield[row['field']];f['tp']+=hit;f['fp']+=extra;f['fn']+=miss;f['exact']+=int(is_exact);f['n']+=1
            details.append({'document_id':row['document_id'],'field':row['field'],'model':model,'gold_json':json_str(gold),
                            'prediction_json':norm_json(pred),'exact':int(is_exact),'tp':hit,'fp':extra,'fn':miss,
                            'gold_present':int(gp),'pred_present':int(pp)})
        p=tp/(tp+fp) if tp+fp else None;r=tp/(tp+fn) if tp+fn else None
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        tpc,fpc,fnc,tnc=(sum_presence[k] for k in ['tp','fp','fn','tn'])
        sens=tpc/(tpc+fnc) if tpc+fnc else None;spec=tnc/(tnc+fpc) if tnc+fpc else None
        denom=math.sqrt((tpc+fpc)*(tpc+fnc)*(tnc+fpc)*(tnc+fnc))
        mcc=(tpc*tnc-fpc*fnc)/denom if denom else None
        per=[]
        for field,c in sorted(byfield.items()):
            den=2*c['tp']+c['fp']+c['fn'];per.append({'model':model,'field':field,'n_verified':c['n'],
                  'micro_f1':2*c['tp']/den if den else None,'exact_match':c['exact']/c['n']})
        valid_macro=[v['micro_f1'] for v in per if v['micro_f1'] is not None]
        aggregate[model]={'verified_fields':len(verified),'verified_documents':len(counts),'tp':tp,'fp':fp,'fn':fn,
                          'precision':p,'recall':r,'micro_f1':f1,'macro_field_f1':statistics.mean(valid_macro) if valid_macro else None,
                          'field_exact_match':exact/len(verified),'unsupported_value_rate':unsupported/(tp+fp) if tp+fp else None,
                          'omission_rate':omitted/(tp+fn) if tp+fn else None,
                          'presence_sensitivity':sens,'presence_specificity':spec,
                          'presence_balanced_accuracy':(sens+spec)/2 if sens is not None and spec is not None else None,
                          'presence_mcc':mcc,'presence_confusion':dict(sum_presence),'per_field':per}
    corrections={}
    for model in ['student','gguf']:
        ct=collections.Counter();audit=[]
        for row,gold in verified:
            golds=collections.Counter(atom_list(gold))
            t=collections.Counter(atom_list(parse_prediction(row,'teacher')))==golds
            s=collections.Counter(atom_list(parse_prediction(row,model)))==golds
            # Missing keys are never counted as correct even for empty gold
            if parse_prediction(row,'teacher') is MISSING:t=False
            if parse_prediction(row,model) is MISSING:s=False
            key=('teacher_correct' if t else 'teacher_wrong')+'__'+('student_correct' if s else 'student_wrong')
            ct[key]+=1
            audit.append({'document_id':row['document_id'],'field':row['field'],'model':model,'outcome':key,'gold_json':json_str(gold),
                          'teacher_json':row['teacher_json'],'student_json':row[f'{model}_json'],'evidence_quote':row['evidence_quote'],'source_page':row['source_page']})
        tw=ct['teacher_wrong__student_correct']+ct['teacher_wrong__student_wrong'];tc=ct['teacher_correct__student_correct']+ct['teacher_correct__student_wrong']
        corrections[model]={'counts':dict(ct),'teacher_error_avoidance_rate':ct['teacher_wrong__student_correct']/tw if tw else None,
                            'student_regression_rate':ct['teacher_correct__student_wrong']/tc if tc else None}
        with (out/f'{model}_teacher_error_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=audit[0].keys());w.writeheader();w.writerows(audit)
    with (out/'verified_field_scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=details[0].keys());w.writeheader();w.writerows(details)
    result={'verified_rows':len(verified),'total_target_rows':2880,'coverage':len(verified)/2880,'verified_documents':len(counts),
            'models':aggregate,'teacher_error_comparison':corrections,
            'methodology':'Verified-only; multiset value-level micro P/R/F1; macro field F1 excludes fields with no positive values; exact match counts missing key as incorrect; presence MCC is a separate binary task. No automatic candidate quote treated as gold.',
            'warning':'Partial verified subsets are not representative unless independently sampled; verify agreement and disagreement rows. Numeric equality is exact to 6 decimals; field-specific unit/alias adjudication must occur in gold review.'}
    (out/'verified_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Scored',len(verified),'verified rows; coverage',f'{100*len(verified)/2880:.1f}%')
    for m,v in aggregate.items():print(m,'F1=',v['micro_f1'],'exact=',v['field_exact_match'])

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',default='.');a.add_argument('--split',default='test',choices=['dev','test']);a.add_argument('--out',default='paper_results/factual_test');a.set_defaults(func=prepare)
    b=sub.add_parser('score');b.add_argument('--reviews',required=True);b.add_argument('--out',default='paper_results/factual_test');b.set_defaults(func=score)
    args=p.parse_args();args.func(args)
if __name__=='__main__':main()
