from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from verification_common import load_json,iter_documents,evidence_for_value

M14='qwen3_14b-otse8k'

def priority(status):
    if 'SEMANTIC_CONFLICT' in status: return 1
    if 'EVIDENCE_ADVANTAGE' in status: return 2
    if 'SUPPORTED' in status: return 3
    if 'WEAK' in status or 'UNSUPPORTED' in status: return 4
    if 'METADATA' in status: return 5
    return 9

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs-root',default='gold_dataset/benchmark_runs'); ap.add_argument('--reports-dir',default='gold_dataset/reports')
    a=ap.parse_args(); root=Path(a.runs_root); out=Path(a.reports_dir); out.mkdir(parents=True,exist_ok=True)
    q=out/'review_queue.csv'
    rows=list(csv.DictReader(q.open(encoding='utf-8-sig'))) if q.exists() else []
    rows.sort(key=lambda r:(priority(r['status']),r['document_id'],r['field']))
    lines=['# OTSE semantic review queue','',f'Field-level cases requiring review: **{len(rows)}**','']
    bydoc={}
    for r in rows: bydoc.setdefault(r['document_id'],[]).append(r)
    for doc,items in bydoc.items():
      lines += [f'## {doc}','']
      for r in items:
        lines += [f"### {r['field']} - {r['status']}",f"- 4B: `{r['value_4b']}`",f"- 14B: `{r['value_14b']}`"]
        ev=r.get('evidence_14b') or r.get('evidence_4b')
        if ev: lines += [f'- Evidence excerpt: {ev[:700]}']
        lines += ['- Decision: [ ] 4B  [ ] 14B  [ ] edit  [ ] ambiguous','']
    # Every mode stays semantic-review-only because association within a source row is not safely reducible to scalar regex checks.
    mode_rows=[]
    for doc,text in iter_documents(root):
      modes=load_json(doc/M14/'modes.json',{}) or {}
      for i,m in enumerate(modes.get('modes',[]) if isinstance(modes,dict) else []):
        checks={}
        for fld in ['application_code','standard','payload_gbps','electrical_interface','fec_type','modulation_format','reach_km','tx_power_min_dbm','tx_power_max_dbm','rx_sensitivity_dbm','cd_tolerance_ps_nm']:
          if fld in m and m[fld] not in (None,'',[]): checks[fld]=evidence_for_value(text,fld,m[fld])['score']
        mode_rows.append({'document_id':doc.name,'mode_index':i+1,'label':m.get('label'),'application_code':m.get('application_code'),'mode_json':json.dumps(m,ensure_ascii=False),'evidence_scores':json.dumps(checks)})
    if mode_rows:
      with (out/'mode_review_queue.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(mode_rows[0])); w.writeheader(); w.writerows(mode_rows)
      lines += ['# Mode-level semantic review','',f'Modes requiring row-level review: **{len(mode_rows)}**','']
      for r in mode_rows:
        lines += [f"## {r['document_id']} - mode {r['mode_index']} ({r['label']})",f"`{r['mode_json']}`",f"Evidence scores: `{r['evidence_scores']}`",'- Decision: [ ] accept  [ ] edit  [ ] reject','']
    (out/'review_queue.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Wrote {out/"review_queue.md"}'); print(f'Field review cases: {len(rows)}; mode review cases: {len(mode_rows)}')
if __name__=='__main__': main()
