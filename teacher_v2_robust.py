"""Resumable training-only teacher extraction; outputs are unreviewed predictions."""
import argparse,csv,json,re,time,hashlib
from pathlib import Path
from collections import Counter

ROOT=Path('data/corpus_v2'); MAN=ROOT/'manifests'; PARSED=ROOT/'parsed'; OUT=ROOT/'teacher_labels'/'train_robust'

def load_rows():
    p=MAN/'v2_split_frozen.csv'
    with p.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if not rows or not {'split','document_id','sha256'}<=set(rows[0]): raise RuntimeError('Invalid split manifest')
    if len({r['document_id'] for r in rows})!=len(rows): raise RuntimeError('Duplicate IDs')
    if not all(r['split']=='train' for r in rows) and len(rows)!=616: raise RuntimeError('Incomplete mixed split')
    return [r for r in rows if r['split']=='train']

def parse_response(text):
    text=text.strip()
    try:return json.loads(text)
    except json.JSONDecodeError: pass
    m=re.search(r'```(?:json)?\s*(.*?)```',text,re.S)
    if m:return json.loads(m.group(1))
    raise ValueError('Response is not a standalone JSON object')

def valid_group(obj,fields):
    if not isinstance(obj,dict):return False,'not_object'
    extra=set(obj)-set(fields)
    if extra:return False,'unexpected_keys:'+','.join(sorted(extra))
    if not obj:return False,'empty_object'
    for k,v in obj.items():
        if isinstance(v,(dict,list)) and (isinstance(v,dict) or any(isinstance(x,(dict,list)) for x in v)):
            return False,'nested_value:'+k
    return True,'ok'

def evidence_flags(source,parsed):
    """Conservative review hints, NOT semantic proof and NOT automatic rejection."""
    text=source.casefold(); flags=[]
    for k,v in parsed.items():
        if v is None or v==[] or v=='':continue
        vals=v if isinstance(v,list) else [v]
        for value in vals:
            s=str(value).strip().casefold()
            if len(s)>2 and s not in text:
                flags.append({'field':k,'value':value,'reason':'not_literal_substring_review_conversion_or_table'})
    return flags

def run(args):
    from openai import OpenAI
    from pipeline.atom_extractor import FIELD_GROUPS,SYSTEM_PROMPT,USER_PROMPT_TEMPLATE,_schema_for_fields,_extract_headings
    rows=load_rows()
    if args.limit:rows=rows[:args.limit]
    if not rows:raise RuntimeError('No training documents')
    client=OpenAI(base_url=args.base_url.rstrip('/')+'/v1',api_key='ollama',timeout=args.timeout)
    OUT.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows,1):
        docid=row['document_id']; path=OUT/(docid+'.json'); source_path=PARSED/(docid+'.txt')
        if not source_path.exists():raise FileNotFoundError('Parse this training PDF first: '+str(source_path))
        source=source_path.read_text(encoding='utf-8')
        if path.exists():
            record=json.loads(path.read_text(encoding='utf-8'))
            if record.get('document_id')!=docid or record.get('source_sha256')!=row['sha256'] or record.get('model')!=args.model:
                raise RuntimeError('Existing output mismatch; inspect before retry: '+str(path))
        else:record={'document_id':docid,'source_sha256':row['sha256'],'model':args.model,'groups':{},'status':'partial'}
        for group,fields in FIELD_GROUPS:
            old=record['groups'].get(group,{})
            if old.get('valid') is True:continue
            prompt=USER_PROMPT_TEMPLATE.format(schema=_schema_for_fields(fields),headings=_extract_headings(source),content=source[:args.max_chars])
            attempts=old.get('attempts',[])
            for j in range(args.retries+1):
                attempt={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
                try:
                    response=client.chat.completions.create(model=args.model,messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':prompt}],temperature=0,max_tokens=args.max_tokens)
                    choice=response.choices[0];raw=choice.message.content or ''
                    attempt.update({'finish_reason':choice.finish_reason,'raw':raw})
                    obj=parse_response(raw)
                    ok,reason=valid_group(obj,fields)
                    if choice.finish_reason=='length':ok=False;reason='truncated'
                    attempt['validation']=reason
                    if ok:
                        record['groups'][group]={'valid':True,'parsed':obj,'requested_fields':list(fields),'finish_reason':choice.finish_reason,'evidence_review_flags':evidence_flags(source,obj),'attempts':attempts+[attempt]}
                        break
                except Exception as e:attempt['validation']='exception:'+str(e)[:500]
                attempts.append(attempt)
            if not record['groups'].get(group,{}).get('valid'):
                record['groups'][group]={'valid':False,'requested_fields':list(fields),'attempts':attempts}
            record['status']='complete' if all(record['groups'].get(g,{}).get('valid') for g,_ in FIELD_GROUPS) else 'partial'
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{i}/{len(rows)} {docid} {record["status"]}',flush=True)
    print('UNREVIEWED teacher predictions; not gold or training-ready.')

def report(args):
    rows=load_rows(); selected=rows[:args.limit] if args.limit else rows
    stats=Counter(); failures=[]
    for r in selected:
        p=OUT/(r['document_id']+'.json')
        if not p.exists():stats['missing_documents']+=1;continue
        d=json.loads(p.read_text(encoding='utf-8'))
        if d.get('status')=='complete':stats['complete_documents']+=1
        else:stats['partial_documents']+=1
        for group,g in d.get('groups',{}).items():
            if g.get('valid'):stats['valid_groups']+=1;stats['evidence_review_flags']+=len(g.get('evidence_review_flags',[]))
            else:stats['invalid_groups']+=1;failures.append([r['document_id'],group])
    print(json.dumps({'summary':dict(stats),'failures':failures},indent=2))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('run');r.add_argument('--model',default='qwen3:14b-otse8k');r.add_argument('--base-url',default='http://127.0.0.1:11434');r.add_argument('--limit',type=int,default=5);r.add_argument('--max-chars',type=int,default=30000);r.add_argument('--max-tokens',type=int,default=3072);r.add_argument('--retries',type=int,default=2);r.add_argument('--timeout',type=int,default=600);r.set_defaults(func=run)
    s=sub.add_parser('report');s.add_argument('--limit',type=int,default=5);s.set_defaults(func=report)
    a=p.parse_args();a.func(a)
if __name__=='__main__':main()
