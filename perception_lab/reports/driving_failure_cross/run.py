"""Describe potential driving-related failures; no validated driving-risk labels."""
import csv,json,hashlib,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'reports/spatial_failure_analysis'))
from reproduce import region
source=ROOT/'reports/mini_evaluation/cases.json';sp=ROOT/'reports/confidence_filter/scores.json'
sc=json.loads(sp.read_text());assert sc['cases_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
rows=[c for c in json.loads(source.read_text()) if c['kind']!='gap_recovery']
D=['0–<10 m','10–<20 m','20–<40 m','≥40 m'];S=['0.25–<0.40','0.40–<0.60','0.60–<0.80','0.80–1.00','无置信度'];K={'false_negative':'漏检','false_positive':'误检','center_error_over_1m':'中心误差>1 m','id_switch':'连续 ID 切换','gap_id_change':'中断后换 ID'}
counts=Counter();joint=Counter();groups=Counter();examples=[]
for c in rows:
 d=c['distance_m'];db=D[next((i for i,h in enumerate([10,20,40]) if d<h),3)]
 s=sc['scores'][c['case_id']];sb=S[4] if s is None else S[next((i for i,h in enumerate([.4,.6,.8]) if s<h),3)]
 group=('行人和骑行者' if c['class_name'] in ['pedestrian','bicycle','motorcycle'] else '护栏和交通锥' if c['class_name'] in ['barrier','traffic_cone'] else '机动车')
 reg=region(*c['center_ego'][:2]);joint[c['module'],group,c['kind'],reg,db,sb]+=1
 for scope in ['全部区域']+(['近前方'] if reg=='front_near' else []):
  counts[scope,c['kind'],db]+=1;counts[scope,c['kind'],sb]+=1
  groups[scope,group,db,sb]+=1
 if reg=='front_near' and (c['kind']=='false_negative' or (s is not None and s>=.8)):
  examples.append(dict(c,confidence=s))
lines=[]
for scope in ['全部区域','近前方']:
 lines+=['## '+scope,'','### 按距离','', '| 失败类型 | '+' | '.join(D)+' | 合计 |','|---|'+'---:|'*5]
 for k,label in K.items():
  vals=[counts[scope,k,d] for d in D];lines+=['| '+label+' | '+' | '.join(map(str,vals+[sum(vals)]))+' |']
 lines+=['','### 按置信度','','| 失败类型 | '+' | '.join(S)+' | 合计 |','|---|'+'---:|'*6]
 for k,label in K.items():
  vals=[counts[scope,k,s] for s in S];lines+=['| '+label+' | '+' | '.join(map(str,vals+[sum(vals)]))+' |']
 lines+=['']
(HERE/'tables.md').write_text('\n'.join(lines))
with (HERE/'joint_counts.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['module','object_group','failure_kind','region','distance_bin','confidence_bin','count']);w.writerows([(*k,n) for k,n in sorted(joint.items())])
with (HERE/'object_group_cross.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['scope','object_group','distance_bin',*S,'total'])
 for scope in ['全部区域','近前方']:
  for group in ['行人和骑行者','机动车','护栏和交通锥']:
   for d in D:
    vals=[groups[scope,group,d,s] for s in S];w.writerow([scope,group,d,*vals,sum(vals)])
assert sum(joint.values())==1910
(HERE/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
(HERE/'verification.json').write_text(json.dumps(dict(cases_sha256=sc['cases_sha256'],scores_sha256=hashlib.sha256(sp.read_bytes()).hexdigest(),total=sum(joint.values()),front_near=sum(n for k,n in joint.items() if k[3]=='front_near'),scope='Original-label baseline max_age=3; records not unique objects; no planner validation'),indent=2)+'\n')
print('\n'.join(lines))
