"""Human case judgments, isolated from immutable evaluation and GT."""
import json,os
from pathlib import Path
from datetime import datetime,timezone
from event_reviews import LOCK,Conflict
LABELS={'original','gt_missing','prediction_correct','gt_error','confirmed_failure','needs_review'}

class LabelStore:
 def __init__(self,path,dataset_id,cases):
  self.path=Path(path);self.dataset_id=dataset_id;self.cases={c['case_id']:c for c in cases}
 def read(self):
  with LOCK:
   data=json.loads(self.path.read_text()) if self.path.exists() else dict(schema_version=1,dataset_id=self.dataset_id,items={},history=[])
   if data['dataset_id']!=self.dataset_id:raise Conflict('案例版本变化，请刷新')
   return data
 def save(self,payload):
  with LOCK:
   data=self.read()
   if payload.get('dataset_id')!=self.dataset_id:raise Conflict('案例版本变化，请刷新')
   ident=payload.get('case_id');label=payload.get('label');note=payload.get('note','')
   if not isinstance(ident,str) or ident not in self.cases:raise ValueError('未知案例')
   original=self.cases[ident]
   if not isinstance(label,str) or label not in LABELS:raise ValueError('未知人工标签')
   if label in {'gt_missing','prediction_correct'} and original['kind']!='false_positive':raise ValueError('检测正确标签仅适用于原始 FP 案例')
   if not isinstance(note,str) or len(note)>1000:raise ValueError('说明不能超过1000字符')
   old=data['items'].get(ident,{'revision':0})
   if type(payload.get('revision')) is not int or payload['revision']!=old['revision']:raise Conflict('其他窗口已更新，请重新读取')
   item=dict(case_id=ident,label=label,note=note,original=original,revision=old['revision']+1,
             updated_at=datetime.now(timezone.utc).isoformat(),source='human_judgment')
   data['items'][ident]=item;data['history'].append(item)
   self.path.parent.mkdir(parents=True,exist_ok=True);temp=self.path.with_suffix('.tmp')
   with temp.open('w') as f:
    json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   os.replace(temp,self.path)
   return item
