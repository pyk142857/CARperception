"""Versioned, atomic local review persistence. No model/evaluation mutations."""
import json,os,threading
from datetime import datetime,timezone
from pathlib import Path
LOCK=threading.RLock()
STATUSES={'pending','confirmed','not_issue','annotation_error','duplicate','investigating','resolved'}
FIELDS=('status','owner','root_cause','action','validation','duplicate_of')

class Conflict(ValueError):pass

class ReviewStore:
    def __init__(self,path,dataset_id,targets):
        self.path=Path(path);self.dataset_id=dataset_id;self.targets=set(targets)
    def read(self):
        with LOCK:
            data=json.loads(self.path.read_text()) if self.path.exists() else dict(
                schema_version=1,dataset_id=self.dataset_id,items={},history=[])
            if data['dataset_id']!=self.dataset_id:raise Conflict('数据版本变化，请刷新')
            return data
    def save(self,payload):
        with LOCK:
            data=self.read()
            if payload.get('dataset_id')!=self.dataset_id:raise Conflict('数据版本变化，请刷新')
            target=payload.get('target_id')
            if target not in self.targets:raise ValueError('未知复核对象')
            old=data['items'].get(target,{'revision':0})
            if payload.get('revision')!=old['revision']:raise Conflict('其他窗口已更新，请重新读取')
            fields={name:payload.get(name,'') for name in FIELDS}
            if any(not isinstance(v,str) or len(v)>4000 for v in fields.values()):
                raise ValueError('复核字段必须为不超过4000字符的文本')
            if fields['status'] not in STATUSES:raise ValueError('无效状态')
            if fields['status']=='resolved' and not fields['validation'].strip():
                raise ValueError('标记已解决前必须填写验证证据')
            if fields['status']=='duplicate':
                if fields['duplicate_of'] not in self.targets or fields['duplicate_of']==target:
                    raise ValueError('重复对象必须是另一个有效事件或分组')
                visited={target};other=fields['duplicate_of']
                while other:
                    if other in visited:raise ValueError('重复对象形成循环')
                    visited.add(other)
                    record=data['items'].get(other,{})
                    other=record.get('duplicate_of') if record.get('status')=='duplicate' else None
            item=dict(fields,target_id=target,revision=old['revision']+1,
                      updated_at=datetime.now(timezone.utc).isoformat())
            data['items'][target]=item;data['history'].append(item.copy())
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temp=self.path.with_suffix('.tmp')
            with temp.open('w') as f:
                json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
            os.replace(temp,self.path)
            return item
