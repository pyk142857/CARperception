"""Cross-tabulate baseline raw failure records, not merged events or unique objects."""
import csv,json,hashlib,math
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
source=ROOT/'reports/mini_evaluation/cases.json';scorepath=ROOT/'reports/confidence_filter/scores.json'
cases=json.loads(source.read_text());sc=json.loads(scorepath.read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==sc['cases_sha256']
assert len({c['case_id'] for c in cases})==len(cases)
assert set(sc['scores'])=={c['case_id'] for c in cases}
distances=['[0,10)','[10,20)','[20,40)','[40,+inf)'];scores=['[0.25,0.40)','[0.40,0.60)','[0.60,0.80)','[0.80,1.00]','N/A (FN)']
counts=Counter();breakdown=Counter();retained=[]
for c in cases:
 if c['kind']=='gap_recovery':continue
 d=c['distance_m'];assert math.isfinite(d) and d>=0
 db=distances[next((i for i,limit in enumerate([10,20,40]) if d<limit),3)]
 s=sc['scores'][c['case_id']]
 if s is None:
  assert c['kind']=='false_negative';sb=scores[-1]
 else:
  assert math.isfinite(s) and .25<=s<=1
  sb=scores[next((i for i,limit in enumerate([.4,.6,.8]) if s<limit),3)]
 counts[db,sb]+=1;breakdown[c['module'],c['kind'],c['class_name'],db,sb]+=1
 retained.append(dict(case_id=c['case_id'],module=c['module'],kind=c['kind'],class_name=c['class_name'],distance_m=d,confidence=s,distance_bin=db,confidence_bin=sb))
def write(name,rows,fields):
 with (HERE/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
rows=[dict(distance_m=d,**{s:counts[d,s] for s in scores},total=sum(counts[d,s] for s in scores)) for d in distances]
rows.append(dict(distance_m='TOTAL',**{s:sum(counts[d,s] for d in distances) for s in scores},total=len(retained)))
assert sum(r['total'] for r in rows[:-1])==len(retained)==1910
write('cross_table.csv',rows,['distance_m']+scores+['total'])
write('cases.csv',retained,list(retained[0]))
fields=['module','kind','class_name','distance_bin','confidence_bin','count']
write('breakdown.csv',[dict(zip(fields,(*k,v))) for k,v in sorted(breakdown.items())],fields)
lines=['| 距离（米） | 0.25–<0.40 | 0.40–<0.60 | 0.60–<0.80 | 0.80–1.00 | 无置信度（FN） | 合计 |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+r['distance_m']+' | '+' | '.join(str(r[s]) for s in scores+['total'])+' |')
(HERE/'table.md').write_text('\n'.join(lines)+'\n')
(HERE/'verification.json').write_text(json.dumps(dict(cases_sha256=sc['cases_sha256'],scores_sha256=hashlib.sha256(scorepath.read_bytes()).hexdigest(),included=len(retained),excluded_gap_recovery=len(cases)-len(retained),null_confidence=sum(c['confidence'] is None for c in retained),counts_by_kind=dict(Counter(c['kind'] for c in retained)),counts_by_module=dict(Counter(c['module'] for c in retained)),scope='all classes; detection + tracking; original labels; max_age=3 baseline; raw case records'),indent=2)+'\n')
print('\n'.join(lines))
