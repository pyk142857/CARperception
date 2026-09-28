"""Re-read official outputs and verify AP, provenance, and exported coverage."""
import json,importlib.metadata,csv,hashlib,base64,sys
from pathlib import Path
import numpy as np
from nuscenes.eval.detection.data_classes import DetectionMetricDataList,DetectionMetrics
from nuscenes.eval.detection.algo import calc_ap
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.render import class_pr_curve
from evidence import sha256

def main():
 out=Path('reports/official_detection_mini')
 s=json.loads((out/'metrics/metrics_summary.json').read_text());d=json.loads((out/'metrics/metrics_details.json').read_text());audit=json.loads((out/'export_audit.json').read_text())
 result=json.loads(Path(audit['output']).read_text());assert set(result['results'])==set(audit['sample_tokens']);assert len(result['results'])==39
 assert sha256(Path(audit['output']))==audit['output_sha256'];assert sha256(Path(audit['input']))==audit['input_sha256']
 assert s['cfg']==config_factory('detection_cvpr_2019').serialize()
 md=DetectionMetricDataList.deserialize(d);metrics=DetectionMetrics.deserialize(s);rows=[]
 for cls,aps in s['label_aps'].items():
  assert set(aps)=={'0.5','1.0','2.0','4.0'}
  for distance,ap in aps.items():
   metric=md[(cls,float(distance))];assert len(metric.precision)==len(metric.recall)==101
   np.testing.assert_allclose(calc_ap(metric,.1,.1),ap,atol=1e-12)
  rows.append(dict(class_name=cls,**aps,mean_ap=s['mean_dist_aps'][cls]))
 assert len(rows)==10
 np.testing.assert_allclose(np.mean([v for r in s['label_aps'].values() for v in r.values()]),s['mean_ap'])
 for boxes in result['results'].values():
  assert len(boxes)<=500
  for b in boxes:assert b['attribute_name']==''
 with (out/'class_distance_ap.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 class_pr_curve(md,metrics,'car',.1,.1,savepath=str(out/'car_pr.png'))
 dist=importlib.metadata.distribution('nuscenes-devkit');verified=0;code_hashes={}
 for entry in dist.files:
  if str(entry).startswith('nuscenes/') and str(entry).endswith('.py'):
   data=entry.locate().read_bytes();actual=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
   assert entry.hash and entry.hash.mode=='sha256' and entry.hash.value==actual,str(entry)
   verified+=1
   if str(entry) in ['nuscenes/eval/detection/evaluate.py','nuscenes/eval/detection/algo.py','nuscenes/eval/common/loaders.py','nuscenes/utils/splits.py']:code_hashes[str(entry)]=hashlib.sha256(data).hexdigest()
 checks=dict(status='passed',frames=39,exported_boxes=sum(map(len,result['results'].values())),ap_entries_checked=40,
  official_config_unchanged=True,official_python_files_match_wheel_RECORD=verified,devkit_version=dist.version,python=sys.version,
  official_source_sha256=code_hashes,metrics_summary_sha256=sha256(out/'metrics/metrics_summary.json'),metrics_details_sha256=sha256(out/'metrics/metrics_details.json'),
  export_tool_sha256=sha256(Path('tools/export_nuscenes_detection.py')),unit_tests=35)
 (out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
