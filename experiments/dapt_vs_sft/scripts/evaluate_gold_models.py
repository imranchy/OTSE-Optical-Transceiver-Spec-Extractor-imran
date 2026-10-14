"""OTSE gold-reference comparison: 4B base/SFT/DAPT/DAPT->SFT; optional Ollama teacher.
Run from original OTSE repository root. Does not train or alter references.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = 'Qwen/Qwen3-4B-Instruct-2507'
MODELS = ('base', 'sft', 'dapt', 'dapt_sft', 'teacher')
ADAPTERS = {
    'sft': 'training_outputs/qwen3-4b-otse-expanded-745/adapter_final',
    'dapt': 'experiments/dapt_vs_sft/outputs/dapt_only/adapter_final',
    'dapt_sft': 'experiments/dapt_vs_sft/outputs/dapt_then_sft/adapter_final',
}

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')
    os.replace(tmp, p)

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def resolve_gold(root, record):
    # Never trust another computer's absolute paths; use the reviewed run ID.
    p = root / 'data/expert_gold/reports/runs' / record['gold_run_id'] / 'specs.json'
    if not p.is_file(): raise FileNotFoundError(f'Gold specs missing: {p}')
    return p

def resolve_pdf(root, record):
    # Hash-verified filename in the local corpus, not an external absolute path.
    p = root / 'data/corpus_v2/raw_pdfs' / Path(record['pdf_path']).name
    if not p.is_file() or sha256(p) != record['pdf_sha256']:
        raise ValueError(f'PDF missing or SHA256 mismatch: {p}')
    return p

def load_tasks(root, manifest, tokenizer, max_chars, context_limit, max_new_tokens):
    import fitz
    from otse_auto.core import GROUPS, messages_for
    from otse_auto.evidence import fit_blocks
    tasks=[]; audit=[]
    seen=set()
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        doc=r['document_id']
        if doc in seen: raise ValueError('Repeated document ID in test manifest: '+doc)
        seen.add(doc)
        if r.get('match_method') != 'pdf_sha256':
            match_warning='HEURISTIC_SOURCE_MATCH_NOT_INDEPENDENTLY_CONFIRMED'
        else: match_warning=None
        pdf=resolve_pdf(root,r); gold_path=resolve_gold(root,r)
        gold=read_json(gold_path)
        if not isinstance(gold,dict): raise ValueError('Gold JSON is not object: '+str(gold_path))
        with fitz.open(pdf) as f: pages=[p.get_text('text',sort=True) for p in f]
        if sum(map(len,pages)) < 80: raise ValueError('Insufficient PDF text: '+str(pdf))
        available=[f for fields in GROUPS.values() for f in fields if f in gold]
        audit.append({'document_id':doc,'pdf':str(pdf),'gold':str(gold_path),'gold_keys':len(gold),
                      'scorable_fields':len(available),'unmapped_gold_keys':sorted(set(gold)-set(available)),
                      'match_method':r.get('match_method'), 'source_match_warning':match_warning})
        for group,fields in GROUPS.items():
            known=[f for f in fields if f in gold]
            if not known: continue
            view=fit_blocks(pages,group,tokenizer,max_chars,context_limit,output_reserve=max_new_tokens)
            msgs=messages_for(group,view['source'])
            prompt=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            nt=len(tokenizer(prompt,add_special_tokens=False)['input_ids'])
            if nt+max_new_tokens>context_limit: raise ValueError('Prompt exceeds context: '+doc+'/'+group)
            tasks.append({'document_id':doc,'group':group,'fields':known,'gold':{k:gold[k] for k in known},
                          'prompt':prompt,'source_sha256':hashlib.sha256(view['source'].encode()).hexdigest(),
                          'pdf_sha256':r['pdf_sha256'],'prompt_tokens':nt})
    if not tasks: raise ValueError('No scorable tasks')
    return tasks,audit

def load_4b(root, model_name):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for local 4B evaluation')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=dtype)
    base=AutoModelForCausalLM.from_pretrained(BASE,quantization_config=q,dtype=dtype,device_map={'':0},attn_implementation='sdpa')
    if model_name!='base':
        p=root/ADAPTERS[model_name]
        cfg=read_json(p/'adapter_config.json')
        if cfg.get('base_model_name_or_path','').rstrip('/')!=BASE:
            raise ValueError('Adapter base checkpoint differs: '+str(p))
        base=PeftModel.from_pretrained(base,str(p),is_trainable=False)
    base.eval();base.config.use_cache=True
    return base

def parse_json(raw):
    try:
        obj=json.loads(raw.strip(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return obj if isinstance(obj,dict) else None
    except (ValueError,TypeError): return None

def infer_hf(model, tokenizer, task, new_tokens):
    import torch
    encoded=tokenizer(task['prompt'],return_tensors='pt',add_special_tokens=False).to('cuda')
    n=encoded['input_ids'].shape[1]
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        out=model.generate(**encoded,do_sample=False,max_new_tokens=new_tokens,
                           pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    tokens=out[0,n:]
    return tokenizer.decode(tokens,skip_special_tokens=True),float(elapsed),int(len(tokens))

def infer_ollama(task, model_name, new_tokens, url):
    # Uses the identical rendered chat prompt as the 4B systems.
    body={'model':model_name,'prompt':task['prompt'],'stream':False,
          'options':{'temperature':0,'num_predict':new_tokens,'num_ctx':16384}}
    req=urllib.request.Request(url+'/api/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter()
    with urllib.request.urlopen(req,timeout=900) as f: obj=json.load(f)
    return obj.get('response',''),time.perf_counter()-start,int(obj.get('eval_count',0))

def score_model(tasks,folder,model_name,scoring):
    by_doc=defaultdict(list);by_group=defaultdict(list);by_field=defaultdict(list)
    details=[];lat=[];schema_fail=0;invalid=0;count=0
    from otse_auto.core import GROUPS
    for task in tasks:
        p=folder/(task['document_id']+'__'+task['group']+'.json')
        if not p.is_file(): raise FileNotFoundError(str(p))
        rec=read_json(p)
        if rec['pdf_sha256']!=task['pdf_sha256'] or rec['source_sha256']!=task['source_sha256']:
            raise ValueError('Stale prediction / changed evidence: '+str(p))
        obj=rec['parsed']; count+=1;lat.append(rec['seconds'])
        if obj is None: invalid+=1
        if obj is None or set(obj)!=set(GROUPS[task['group']]):schema_fail+=1
        for field in task['fields']:
            pred=obj.get(field,scoring.MISSING) if isinstance(obj,dict) else scoring.MISSING
            for mode in ('literal','canonical'):
                d=scoring.score_one({'field':field,'gold_value':task['gold'][field]},pred,mode=mode)
                by_doc[(mode,task['document_id'])].append(d['counts'])
                by_group[(mode,task['group'])].append(d['counts'])
                by_field[(mode,field)].append(d['counts'])
                details.append({'model':model_name,'document_id':task['document_id'],'group':task['group'],
                                'field':field,'metric_mode':mode,'gold':task['gold'][field],
                                'prediction':None if pred is scoring.MISSING else pred,
                                'tp':d['counts']['tp'],'fp':d['counts']['fp'],'fn':d['counts']['fn'],
                                'exact_match':d['exact_match'],'schema_keys_valid':bool(obj is not None and set(obj)==set(GROUPS[task['group']]))})
    result={'model':model_name,'documents':len({t['document_id'] for t in tasks}),'tasks':count,
            'scored_field_decisions':sum(len(t['fields']) for t in tasks),'malformed_json_tasks':invalid,
            'schema_key_failure_tasks':schema_fail,'latency_seconds_median_per_group':statistics.median(lat),
            'latency_seconds_mean_per_group':statistics.mean(lat),
            'warning':'Field/value matching against reviewed flat JSON; not evidence-span-verified or mode-aware extractive F1.'}
    for mode in ('literal','canonical'):
        counts=scoring.add_counts(c for (m,_),rows in by_doc.items() if m==mode for c in rows)
        result[mode]=scoring.metrics(counts)
        result[mode+'_by_group']={g:scoring.metrics(scoring.add_counts(v)) for (m,g),v in by_group.items() if m==mode}
        result[mode+'_by_field']={f:scoring.metrics(scoring.add_counts(v)) for (m,f),v in by_field.items() if m==mode}
    return result,details

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--models',nargs='+',choices=MODELS,default=['base','sft','dapt','dapt_sft'])
    ap.add_argument('--teacher-ollama-model',default='',help='Ollama tag; required only for --models teacher')
    ap.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    ap.add_argument('--max-chars',type=int,default=7000)
    ap.add_argument('--context-limit',type=int,default=8192)
    ap.add_argument('--max-new-tokens',type=int,default=1024)
    ap.add_argument('--limit-documents',type=int,default=0,help='Development subset; 0 evaluates all')
    args=ap.parse_args()
    root=Path(args.root).resolve()
    if 'teacher' in args.models and not args.teacher_ollama_model: ap.error('Provide --teacher-ollama-model for teacher')
    from transformers import AutoTokenizer
    from otse_auto import scoring
    from otse_auto.core import GROUPS
    tok=AutoTokenizer.from_pretrained(BASE,use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    manifest=root/'experiments/dapt_vs_sft/results/gold_benchmark/gold_test_manifest.jsonl'
    if not manifest.is_file(): raise FileNotFoundError(str(manifest))
    if args.limit_documents:
        lines=[json.loads(x) for x in manifest.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
        subset=lines[:args.limit_documents]
        tmp=root/'experiments/dapt_vs_sft/results/gold_benchmark/_subset_manifest.jsonl'
        tmp.write_text(''.join(json.dumps(r)+'\n' for r in subset),encoding='utf-8');manifest=tmp
    tasks,audit=load_tasks(root,manifest,tok,args.max_chars,args.context_limit,args.max_new_tokens)
    out=root/'experiments/dapt_vs_sft/results/gold_evaluation'
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'coverage_audit.json',{'documents':audit,'groups':GROUPS,
               'source_match_warning':'Text-heuristic PDF matching must be independently confirmed before final claims',
               'gold_absence_policy':'Only keys explicitly present in gold JSON are scored; omitted keys are not assumed null',
               'metric_warning':'Literal/canonical value matching; not span-supported mode-aware extractive F1'})
    all_results={}
    for model_name in args.models:
        folder=out/'predictions'/model_name;folder.mkdir(parents=True,exist_ok=True)
        pending=[]
        for task in tasks:
            path=folder/(task['document_id']+'__'+task['group']+'.json')
            if path.is_file():
                rec=read_json(path)
                if (rec.get('pdf_sha256')==task['pdf_sha256'] and
                    rec.get('source_sha256')==task['source_sha256'] and
                    rec.get('max_new_tokens')==args.max_new_tokens and
                    rec.get('model')==model_name and
                    rec.get('teacher_tag')==(args.teacher_ollama_model if model_name=='teacher' else '')):
                    continue
            pending.append((task,path))
        print(f'\n{model_name}: {len(pending)} pending of {len(tasks)} tasks',flush=True)
        model=None
        try:
            if pending and model_name!='teacher':model=load_4b(root,model_name)
            for i,(task,path) in enumerate(pending,1):
                if model_name=='teacher':
                    raw,seconds,n_tokens=infer_ollama(task,args.teacher_ollama_model,args.max_new_tokens,args.ollama_url)
                else:
                    raw,seconds,n_tokens=infer_hf(model,tok,task,args.max_new_tokens)
                parsed=parse_json(raw)
                write_json(path,{'model':model_name,'teacher_tag':args.teacher_ollama_model if model_name=='teacher' else '',
                    'document_id':task['document_id'],'group':task['group'],'pdf_sha256':task['pdf_sha256'],
                    'source_sha256':task['source_sha256'],'max_new_tokens':args.max_new_tokens,
                    'prompt_tokens':task['prompt_tokens'],'output_tokens':n_tokens,'seconds':seconds,
                    'parsed':parsed,'raw_response':raw})
                print(f'  {model_name} {i}/{len(pending)} {task["document_id"]} {task["group"]} {seconds:.1f}s',flush=True)
        finally:
            if model is not None:
                del model;gc.collect()
                import torch
                torch.cuda.empty_cache()
        result,details=score_model(tasks,folder,model_name,scoring)
        write_json(out/(model_name+'_metrics.json'),result)
        details_path=out/(model_name+'_field_details.jsonl')
        with details_path.open('w',encoding='utf-8') as f:
            for r in details:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n')
        all_results[model_name]=result
        write_json(out/'comparison_summary.json',all_results)
        print(f'{model_name}: literal F1={result["literal"]["micro_f1"]}; canonical F1={result["canonical"]["micro_f1"]}',flush=True)
    print('\nCompleted. Results:',out,flush=True)

if __name__=='__main__': main()
