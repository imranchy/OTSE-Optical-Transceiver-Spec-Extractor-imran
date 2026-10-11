from __future__ import annotations
import argparse,csv,json,re
from collections import Counter
from pathlib import Path
from verification_common import *

M4="qwen3_4b-otse8k"; M14="qwen3_14b-otse8k"
HIGH_VALUE_FIELDS={
 'vendor','model','form_factor','standards_claimed','wavelength_min_nm','wavelength_max_nm','wavelength_center_nm','wavelength_band',
 'frequency_min_thz','frequency_max_thz','channel_spacing_ghz','connector_type','fiber_type','temp_min_c','temp_max_c','power_consumption_w',
 'host_interface_name','media_interface_name','standards_code','baud_rate_gbaud','bit_rate_gbps','modulation_format','reach_km','tx_power_min_dbm',
 'tx_power_max_dbm','rx_sensitivity_dbm','rx_overload_dbm','rx_osnr_db','cd_tolerance_ps_nm','pdl_db','pmd_tolerance_ps','fec_type',
 'fec_overhead_percent','pre_fec_ber_threshold','extinction_ratio_db','los_assert_dbm','los_deassert_dbm'
}

def classify(field,cmp,e4,e14,valid4,valid14):
    if field in METADATA_FIELDS: return 'REVIEW_METADATA'
    if cmp=='BOTH_EMPTY': return 'BOTH_EMPTY'
    if cmp=='AGREE' and valid4 and valid14 and e4['strong'] and e14['strong']:
        return 'AUTO_ACCEPT'
    if cmp=='AGREE': return 'REVIEW_AGREE_WEAK_EVIDENCE'
    if cmp=='14B_ONLY' and valid14 and e14['strong']: return 'REVIEW_14B_SUPPORTED'
    if cmp=='4B_ONLY' and valid4 and e4['strong']: return 'REVIEW_4B_SUPPORTED'
    if cmp=='DISAGREE' and e4['strong'] and not e14['strong']: return 'REVIEW_4B_EVIDENCE_ADVANTAGE'
    if cmp=='DISAGREE' and e14['strong'] and not e4['strong']: return 'REVIEW_14B_EVIDENCE_ADVANTAGE'
    if cmp=='DISAGREE' and e4['strong'] and e14['strong']: return 'REVIEW_SEMANTIC_CONFLICT'
    return 'REVIEW_UNSUPPORTED_OR_AMBIGUOUS'

def source_candidates(text:str):
    out=[]
    pats=[
      ('reach_km',r'(?i)(?:reach|distance|link length|transmission)[^\n]{0,90}?(-?\d+(?:\.\d+)?)\s*km'),
      ('wavelength_nm',r'(?i)(-?\d+(?:\.\d+)?)\s*nm'),
      ('frequency_thz',r'(?i)(\d+(?:\.\d+)?)\s*thz'),
      ('channel_spacing_ghz',r'(?i)(?:spacing|grid)[^\n]{0,80}?(\d+(?:\.\d+)?)\s*ghz'),
      ('bit_rate_gbps',r'(?i)(\d+(?:\.\d+)?)\s*(?:gb/s|gbps|gbit/s)'),
      ('baud_rate_gbaud',r'(?i)(\d+(?:\.\d+)?)\s*(?:gbaud|gbd)'),
      ('temperature_c',r'(?i)(-?\d+(?:\.\d+)?)\s*°?c'),
      ('power_dbm',r'(?i)(-?\d+(?:\.\d+)?)\s*dbm'),
    ]
    for field,pat in pats:
      vals=[]
      for m in re.finditer(pat,text):
        v=float(m.group(1)); vals.append(v)
      vals=sorted(set(vals))
      if vals: out.append({'candidate_type':field,'values':vals[:40]})
    terms={
      'fec':['CFEC','oFEC','RS-FEC','RS(528,514)','RS(544,514)'],
      'modulation':['DP-16QAM','16QAM','PAM4','NRZ','QPSK'],
      'form_factor':['SFP+','SFP28','QSFP+','QSFP28','QSFP-DD','OSFP'],
    }
    low=text.lower()
    for typ,items in terms.items():
      found=[x for x in items if x.lower() in low]
      if found: out.append({'candidate_type':typ,'values':found})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs-root',default='gold_dataset/benchmark_runs'); ap.add_argument('--reports-dir',default='gold_dataset/reports')
    a=ap.parse_args(); root=Path(a.runs_root); out=Path(a.reports_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; missing=[]; statuses=Counter()
    for doc,text in iter_documents(root):
      s4=load_json(doc/M4/'specs.json',{}) or {}; s14=load_json(doc/M14/'specs.json',{}) or {}
      m4=metrics_for(doc,M4); m14=metrics_for(doc,M14); valid4=bool(m4.get('atomic_validation_passed')); valid14=bool(m14.get('atomic_validation_passed'))
      for field in sorted(set(s4)|set(s14)):
        v4=s4.get(field); v14=s14.get(field); cmp=compare_values(v4,v14)
        e4=evidence_for_value(text,field,v4); e14=evidence_for_value(text,field,v14)
        status=classify(field,cmp,e4,e14,valid4,valid14); statuses[status]+=1
        if status=='BOTH_EMPTY' and field not in HIGH_VALUE_FIELDS: continue
        rows.append({
          'document_id':doc.name,'field':field,'comparison':cmp,'status':status,
          'value_4b':json.dumps(v4,ensure_ascii=False),'value_14b':json.dumps(v14,ensure_ascii=False),
          'evidence_score_4b':e4['score'],'evidence_score_14b':e14['score'],
          'evidence_4b':' | '.join(e4['excerpts']),'evidence_14b':' | '.join(e14['excerpts']),
          'atomic_valid_4b':valid4,'atomic_valid_14b':valid14,
        })
      # Surface source candidates for high-value values that can guide missing-field checks.
      missing.append({'document_id':doc.name,'source_candidates':source_candidates(text)})
    cols=list(rows[0]) if rows else []
    with (out/'verification_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
      w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    (out/'source_candidates.json').write_text(json.dumps(missing,indent=2,ensure_ascii=False),encoding='utf-8')
    auto=[r for r in rows if r['status']=='AUTO_ACCEPT']; review=[r for r in rows if r['status'].startswith('REVIEW')]
    for name,data in [('auto_accept.csv',auto),('review_queue.csv',review)]:
      with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(data)
    (out/'verification_summary.json').write_text(json.dumps({'status_counts':dict(statuses),'auto_accept':len(auto),'review':len(review)},indent=2),encoding='utf-8')
    print(json.dumps({'status_counts':dict(statuses),'auto_accept':len(auto),'review':len(review)},indent=2)); print(f'Wrote reports to {out}')
if __name__=='__main__': main()
