from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
from verification_common import *

M4="qwen3_4b-otse8k"; M14="qwen3_14b-otse8k"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs-root',default='gold_dataset/benchmark_runs')
    ap.add_argument('--reports-dir',default='gold_dataset/reports')
    a=ap.parse_args(); root=Path(a.runs_root); out=Path(a.reports_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; counts=Counter()
    for doc,text in iter_documents(root):
        s4=load_json(doc/M4/'specs.json',{}) or {}; s14=load_json(doc/M14/'specs.json',{}) or {}
        m4=metrics_for(doc,M4); m14=metrics_for(doc,M14)
        for field in sorted(set(s4)|set(s14)):
            v4=s4.get(field); v14=s14.get(field); cmp=compare_values(v4,v14); counts[cmp]+=1
            rows.append({
                'document_id':doc.name,'field':field,'comparison':cmp,
                'value_4b':json.dumps(v4,ensure_ascii=False),'value_14b':json.dumps(v14,ensure_ascii=False),
                'atomic_valid_4b':m4.get('atomic_validation_passed'),'atomic_valid_14b':m14.get('atomic_validation_passed'),
            })
    path=out/'field_comparison.csv'
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []); w.writeheader(); w.writerows(rows)
    (out/'comparison_summary.json').write_text(json.dumps({'documents':len({r['document_id'] for r in rows}),'fields_compared':len(rows),'counts':dict(counts)},indent=2),encoding='utf-8')
    print(json.dumps(dict(counts),indent=2)); print(f'Wrote {path}')
if __name__=='__main__': main()
