import csv,json,pathlib,collections
p=pathlib.Path(__file__).resolve().parent/'gold_review.csv'
rows=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
verified=[r for r in rows if r.get('review_status')=='verified']
print(json.dumps({'documents':len(set(r['document_id'] for r in rows)),'total_fields':len(rows),'verified':len(verified),'pending':len(rows)-len(verified),'documents_with_verified':len(set(r['document_id'] for r in verified)),'full_gold_verified':len(verified)==len(rows)},indent=2))
if len(verified)<len(rows):
 print('STOP: Full-test factual F1 cannot be calculated from unverified references.')
