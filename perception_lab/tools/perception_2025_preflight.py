"""Read-only inventory of COCO and original mini point labels; no inference claims."""
import json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1];out=p/'reports/perception_2025_20261013';(out/'inputs').mkdir(parents=True,exist_ok=True)
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
a=p/'data/coco/annotations/instances_val2017.json';c=json.loads(a.read_text());rows=[]
for x in c['images']:
 f=p/'data/coco/images/val2017'/x['file_name']
 if not f.exists():f=p/'data/coco/val2017'/x['file_name']
 if not f.exists():raise FileNotFoundError(f)
 rows.append(dict(id=x['id'],file=str(f),bytes=f.stat().st_size,sha256=sha(f)))
assert len(rows)==5000 and len({x['id'] for x in rows})==5000
(out/'inputs/coco_files.json').write_text(json.dumps(rows,indent=2)+'\n')
d=p/'data/nuscenes';labels=json.loads((d/'v1.0-mini/lidarseg.json').read_text());sd={x['token']:x for x in json.loads((d/'v1.0-mini/sample_data.json').read_text())};l=[]
for x in labels:
 pt=d/sd[x['token']]['filename'];gt=d/x['filename'];n=pt.stat().st_size//20;v=np.fromfile(gt,dtype=np.uint8);assert len(v)==n and v.max()<32
 l.append(dict(sample_data_token=x['token'],point_file=str(pt),label_file=str(gt),points=n,point_sha256=sha(pt),label_sha256=sha(gt)))
(out/'inputs/lidarseg_files.json').write_text(json.dumps(l,indent=2)+'\n')
summary=dict(status='passed',source='new_inventory_not_inference',coco_images=len(rows),coco_annotation_sha256=sha(a),lidarseg_frames=len(l),lidarseg_points=sum(x['points'] for x in l),scene_inputs='inputs/summary.json',training_prohibited=True)
(out/'preflight.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
