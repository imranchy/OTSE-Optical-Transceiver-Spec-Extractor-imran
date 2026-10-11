"""Value-matching functions reused from the user-provided frozen scorer.
No historical test references or predictions are loaded by this module.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import html
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent
MODELS=('teacher','student','gguf')
TEXT_FIELDS={'vendor','model','form_factor','standards_claimed','wavelength_band','connector_type','fiber_type','host_interface_name','host_interface_id_hex','media_interface_name','media_interface_id_hex','frame','standards_code','modulation_format','fec_type'}
BER_FIELDS={'pre_fec_ber_threshold','post_fec_ber_target'}
INT_FIELDS={'channel_total','host_interface_id','media_interface_id'}
MISSING=object()
COUNTS=('n','gold_present','tp','fp','fn','correct','positive_correct','missing','malformed','responded','presence_tp','presence_fp','presence_fn','presence_tn')

def load_json(path:Path)->Any:
    return json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Non-finite JSON number: '+x)))

def dump(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def write_csv(path:Path,rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('',encoding='utf-8'); return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)

def safe_div(a:float,b:float)->float|None:
    return a/b if b else None

def text_base(s:str)->str:
    s=unicodedata.normalize('NFKC',s).casefold().strip()
    for ch in ('\u2010','\u2011','\u2012','\u2013','\u2014','\u2212'):
        s=s.replace(ch,'-')
    s=s.replace('\u00d7','x')
    return re.sub(r'\s+',' ',s)

def normalize_text(field:str,s:str,aliases:list[str]|None=None,gold_value:Any=None)->str:
    t=text_base(s)
    if aliases and isinstance(gold_value,str) and t in {text_base(x) for x in aliases}:
        t=text_base(gold_value)
    if field=='vendor':
        compact=re.sub(r'[^a-z0-9]','',t)
        for start,canonical in [('linkpp','link-pp'),('smartoptics','smartoptics'),('qsfptek','qsfptek'),('fluxlight','fluxlight'),('xgigacommunication','xgiga'),('atgbics','atgbics'),('prolabs','prolabs'),('peakoptical','peak optical')]:
            if compact.startswith(start):return canonical
        return t
    if field=='fiber_type':
        z=re.sub(r'[^a-z0-9/]','',t)
        if z in {'sm','smf','sm/smf','singlemode','singlemodefiber','singlemodefibre'}:return 'smf'
        if z in {'mm','mmf','mm/mmf','multimode','multimodefiber','multimodefibre'}:return 'mmf'
        if z in {'copper','coppercable'}:return 'copper'
        return re.sub(r'\s+','',t)
    if field=='connector_type':
        z=re.sub(r'\s+','',t)
        if re.fullmatch(r'(?:[1248]x)?(?:dual|duplex|simplex)?lc(?:/upc|/apc)?(?:receptacle|connector)?',z):return 'lc'
        if z in {'mpo','mtp','mtp/mpo','mpo/mtp','mpo-12','mpo12','mpo-16','mpo16','mpo-24','mpo24','mpo-12/apc','mpo/apc','dualmpo-12','2xmpo12'}:return 'mpo'
        return z
    if field in {'host_interface_id_hex','media_interface_id_hex'}:
        if re.fullmatch(r'0x[0-9a-f]+',t):return hex(int(t,16))
        return t
    if field=='form_factor':
        t=re.sub(r'\s+','',t)
        return {'qsfpdd':'qsfp-dd','sfpdd':'sfp-dd','qsfp-dd800':'qsfp-dd800'}.get(t,t)
    if field=='fec_type':
        t=re.sub(r'\s+','',t)
        return {'kp4fec':'kp4','rsfec':'rs-fec','o-fec':'ofec','scfec':'sc-fec'}.get(t,t)
    if field=='wavelength_band':
        t=re.sub(r'[\s-]','',t)
        return {'gray':'grey','lanwdm':'lanwdm','cband':'cband','oband':'oband','lband':'lband'}.get(t,t)
    if field=='frame':
        t=t.replace('gbps','g').replace('gb/s','g').replace('gbits/s','g')
        return re.sub(r'\s+','',t)
    # Formatting only: no inference that LR == LR1, generic FEC == a specific code,
    # PAM4 == NRZ, or a protocol is a host/media interface code.
    if field in {'standards_claimed','standards_code','host_interface_name','media_interface_name'}:
        t=re.sub(r'\bieee\s*p(?=802\.)','ieee ',t)
        t=re.sub(r'itu[- ]?t\s+(?=694\.)','itu-t g.',t)
        t=t.replace('lan-wdm','lanwdm').replace('lan wdm','lanwdm')
        return re.sub(r'[\s_-]','',t)
    if field=='modulation_format':return re.sub(r'[\s-]','',t)
    return t

def numeric(field:str,x:Any)->float:
    if isinstance(x,bool):raise ValueError('Boolean is not numeric evidence')
    if isinstance(x,str):
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',x.strip()):raise ValueError('Not a numeric literal')
        x=float(x)
    if not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('Not a finite number')
    if field in INT_FIELDS and float(x)!=int(x):raise ValueError('Not an integer')
    return float(x)

def equal_value(field:str,a:Any,b:Any,row:dict,mode:str='canonical')->bool:
    if isinstance(a,dict) or isinstance(b,dict):return False
    if field in TEXT_FIELDS:
        if not isinstance(a,str) or not isinstance(b,str):return False
        if mode=='literal':return text_base(a)==text_base(b)
        return normalize_text(field,a,row.get('accepted_aliases'),row.get('gold_value'))==normalize_text(field,b,row.get('accepted_aliases'),row.get('gold_value'))
    try:a,b=numeric(field,a),numeric(field,b)
    except ValueError:return False
    if field=='cd_tolerance_ps_nm' and row.get('symmetric_tolerance'):a,b=abs(a),abs(b)
    # BER has ZERO absolute tolerance: 0 cannot match any positive BER.
    return math.isclose(a,b,rel_tol=1e-6,abs_tol=0.0 if field in BER_FIELDS else 1e-12)

def value_list(field:str,value:Any,row:dict,mode:str='canonical')->tuple[list[Any],bool]:
    if value is None:return [],False
    if isinstance(value,str) and text_base(value) in {'','null','none','n/a','not specified','not stated','unknown'}:return [],False
    if field=='fec_type' and row.get('explicit_no_fec') and isinstance(value,str) and text_base(value) in {'no fec','without fec'}:return [],False
    values=value if isinstance(value,list) else [value]
    out=[];bad=False
    for x in values:
        if x is None:continue # null elements make raw schema invalid, but assert no optical value
        if isinstance(x,str) and not x.strip():continue
        if field=='fec_type' and row.get('explicit_no_fec') and isinstance(x,str) and text_base(x) in {'no fec','without fec'}:continue
        valid=isinstance(x,str) if field in TEXT_FIELDS else True
        if field not in TEXT_FIELDS:
            try:numeric(field,x)
            except ValueError:valid=False
        if not valid:
            bad=True
            # One explicit wrong-shaped item is one unmatched claim; not a correct abstention.
            x={'invalid_prediction_item':x}
        if not any(equal_value(field,x,y,row,mode) or (isinstance(x,dict) and x==y) for y in out):out.append(x)
    return out,bad

def maximum_matches(field:str,pred:list,truth:list,row:dict,mode:str)->list[tuple[int,int]]:
    """Maximum bipartite matching, so one prediction cannot satisfy two references."""
    edges=[[j for j,g in enumerate(truth) if equal_value(field,p,g,row,mode)] for p in pred]
    matched={}
    def augment(i:int,seen:set)->bool:
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in matched or augment(matched[j],seen):matched[j]=i;return True
        return False
    for i in range(len(pred)):augment(i,set())
    return [(i,j) for j,i in matched.items()]

def score_one(row:dict,prediction:Any,mode:str='canonical')->dict:
    field=row['field'];g,_=value_list(field,row['gold_value'],row,mode)
    missing=prediction is MISSING
    p,bad=([],False) if missing else value_list(field,prediction,row,mode)
    matches=maximum_matches(field,p,g,row,mode)
    tp=len(matches);fp=len(p)-tp;fn=len(g)-tp
    correct=not missing and not bad and fp==0 and fn==0
    responded=not missing and not bad
    c=dict.fromkeys(COUNTS,0)
    c.update(n=1,gold_present=int(bool(g)),tp=tp,fp=fp,fn=fn,correct=int(correct),positive_correct=int(correct and bool(g)),missing=int(missing),malformed=int(bad),responded=int(responded))
    if responded:c['presence_'+('tp' if g and p else 'fn' if g else 'fp' if p else 'tn')]=1
    return {'counts':c,'prediction_missing':missing,'prediction_malformed':bad,'prediction':None if missing else prediction,
            'reference_values':g,'predicted_values':p,'matched_values':[g[j] for i,j in matches],
            'unmatched_predicted_values':[x for i,x in enumerate(p) if i not in {q for q,j in matches}],
            'omitted_reference_values':[x for j,x in enumerate(g) if j not in {r for i,r in matches}],
            'exact_match':correct}

def add_counts(rows)->dict:
    c=dict.fromkeys(COUNTS,0)
    for r in rows:
        for k in COUNTS:c[k]+=r[k]
    return c

def metrics(c:dict)->dict:
    tp,fp,fn=c['tp'],c['fp'],c['fn'];a,b,d,e=[c[k] for k in ('presence_tp','presence_fp','presence_fn','presence_tn')]
    sensitivity=safe_div(a,a+d);specificity=safe_div(e,e+b)
    den=(a+b)*(a+d)*(e+b)*(e+d)
    return {**c,'precision':safe_div(tp,tp+fp),'recall':safe_div(tp,tp+fn),'micro_f1':safe_div(2*tp,2*tp+fp+fn),
            'exact_match_accuracy':safe_div(c['correct'],c['n']),'positive_field_exact_match_accuracy':safe_div(c['positive_correct'],c['gold_present']),
            'unsupported_or_mismatched_value_rate':safe_div(fp,tp+fp),'omission_rate':safe_div(fn,tp+fn),
            'presence_response_coverage':safe_div(c['responded'],c['n']),
            'presence_accuracy_on_resolved_responses':safe_div(a+e,a+b+d+e),'presence_sensitivity':sensitivity,'presence_specificity':specificity,
            'presence_balanced_accuracy':(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            'presence_mcc':(a*e-b*d)/math.sqrt(den) if den else None,
            'presence_accuracy_counting_missing_or_malformed_as_errors':safe_div(a+e,c['n'])}
