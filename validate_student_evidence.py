"""OTSE conservative evidence-validation development experiment.
Never modifies baseline predictions or references. Run from OTSE repository root.
"""
import argparse, copy, csv, json, re, statistics
from pathlib import Path
from otse_auto import scoring
from otse_auto.core import read_jsonl, write_json

FIELDS = {'temp_min_c','temp_max_c','host_interface_name','host_interface_id','host_interface_id_hex',
          'frequency_min_thz','frequency_max_thz','wavelength_center_nm','standards_claimed'}
FORM_FACTORS = {'sfp','sfp+','sfp28','sfp56','xfp','qsfp','qsfp+','qsfp28','qsfp-dd','osfp','cfp','cfp2','cfp2-dco'}

def evidence(row):
    s = next((m.get('content','') for m in row['messages'] if m.get('role')=='user'), '')
    return s.split('## Datasheet content\n',1)[-1] if '## Datasheet content\n' in s else s

def values(v):
    return v if isinstance(v,list) else [v]

def check(field, value, source):
    """Return (action, reason). No gold labels consulted here."""
    if value is None or value == []: return 'keep','null_or_empty'
    if field == 'host_interface_name':
        if all(isinstance(v,str) and v.strip().lower() in FORM_FACTORS for v in values(value)):
            return 'null','form_factor_not_electrical_host_interface'
    if field in {'host_interface_id','host_interface_id_hex'}:
        if all((isinstance(v,str) and re.fullmatch(r'0x0+',v.strip(),re.I)) or (isinstance(v,(int,float)) and not isinstance(v,bool) and v==0) for v in values(value)):
            return 'null','all_zero_interface_identifier_requires_manual_confirmation'
    if field in {'frequency_min_thz','frequency_max_thz'}:
        if not re.search(r'\b(?:THz|GHz)\b',source,re.I):
            return 'null','no_explicit_frequency_units_in_evidence'
    if field in {'temp_min_c','temp_max_c'}:
        # Diagnostic only: a table can have multiple variants, so never automatically suppress.
        return 'flag','temperature_operating_vs_storage_needs_row_review'
    if field == 'wavelength_center_nm':
        return 'flag','check_exact_wavelength_precision_and_variant_alignment'
    if field == 'standards_claimed':
        return 'flag','verify_each_standard_exactly_against_evidence'
    return 'keep','no_conservative_rule_triggered'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dev',default='training_outputs/otse-auto-source-all-v1/dataset/dev.jsonl')
    p.add_argument('--predictions',default='training_outputs/otse-auto-source-all-v1/predictions/gguf_baseline')
    p.add_argument('--out',default='training_outputs/otse-evidence-validator-dev')
    a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(Path(a.dev)); changes=[]; tally_before=[]; tally_after=[]; missing=[]; touched=0
    for row in rows:
        src=evidence(row)
        name=row['document_id']+'__'+row['field_group']+'.json'
        path=Path(a.predictions)/name
        if not path.is_file(): missing.append(name); continue
        original=json.loads(path.read_text(encoding='utf-8-sig'))
        if original.get('source_view_sha256')!=row['source_view_sha256']: raise ValueError('Source view mismatch: '+name)
        before=original.get('parsed')
        if not isinstance(before,dict): raise ValueError('Malformed baseline: '+name)
        after=copy.deepcopy(before)
        for field in row['known_fields']:
            if field not in FIELDS: continue
            value=before.get(field)
            action,reason=check(field,value,src)
            if action=='null':
                after[field]=None; touched+=1
            if action!='keep':
                changes.append({'document_id':row['document_id'],'field_group':row['field_group'],'field':field,
                    'action':action,'reason':reason,'before':json.dumps(value,ensure_ascii=False),
                    'after':json.dumps(after.get(field),ensure_ascii=False),'silver_reference':json.dumps(row['target_values'][field],ensure_ascii=False)})
        result={k:original[k] for k in ('document_id','field_group','split','source_view_sha256')}
        result.update({'parsed':after,'schema_keys_ok':original.get('schema_keys_ok'),
                       'validator':'conservative_dev_v1','baseline_file':str(path)})
        write_json(out/'predictions'/name,result)
        for field in row['known_fields']:
            gold={'field':field,'gold_value':row['target_values'][field]}
            tally_before.append(scoring.score_one(gold,before.get(field,scoring.MISSING))['counts'])
            tally_after.append(scoring.score_one(gold,after.get(field,scoring.MISSING))['counts'])
    if missing: raise FileNotFoundError('Missing baseline predictions: '+', '.join(missing[:5]))
    before=scoring.metrics(scoring.add_counts(tally_before))
    after=scoring.metrics(scoring.add_counts(tally_after))
    report={'scope':'EXPLORATORY_DEVELOPMENT_ONLY_NOT_HOLDOUT','documents':len(set(r['document_id'] for r in rows)),
            'tasks':len(rows),'known_field_decisions':len(tally_before),'auto_null_actions':touched,
            'flagged_or_modified':len(changes),'before':before,'after':after,
            'delta_micro_f1':(after['micro_f1']-before['micro_f1']) if before['micro_f1'] is not None and after['micro_f1'] is not None else None,
            'caution':'Rules were selected after viewing development errors. Never claim this as held-out improvement. Manual evidence review required.'}
    write_json(out/'report.json',report)
    with (out/'audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['document_id','field_group','field','action','reason','before','after','silver_reference'])
        writer.writeheader();writer.writerows(changes)
    print(json.dumps({'tasks':report['tasks'],'auto_null_actions':touched,'flags_or_changes':len(changes),
        'baseline_f1':before['micro_f1'],'validated_f1':after['micro_f1'],'delta':report['delta_micro_f1'],
        'report':str(out/'report.json')},indent=2))
if __name__=='__main__': main()
