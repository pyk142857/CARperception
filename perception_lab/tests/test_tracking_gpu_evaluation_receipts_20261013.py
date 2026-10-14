"""Do not promote stale or incomplete official-evaluation receipts."""
import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from tracking_gpu_evaluate_20261013 import sync_model_status,SPECS,sha

def fixture(tmp_path):
 folder=tmp_path/'frozen_reference';folder.mkdir();rows=[]
 for split,(frames,*_) in SPECS.items():
  base=folder/split;(base/'official_eval').mkdir(parents=True)
  (base/'tracking.json').write_text('{"results":{}}')
  (base/'official_eval/metrics_summary.json').write_text('{"amota":0.5}')
  rows.append(dict(split=split,frames=frames,tracking_sha256=sha(base/'tracking.json'),metrics_sha256=sha(base/'official_eval/metrics_summary.json')))
 (folder/'evaluation_status.json').write_text(json.dumps(dict(status='passed',rows=rows)))
 for p in [folder/'status.json',tmp_path/'status.json']:p.write_text(json.dumps(dict(inference='passed',status='partial',official_evaluation='pending')))
 return folder

def test_bound_receipt_and_new_running_state(tmp_path):
 folder=fixture(tmp_path);sync_model_status('grae_gpu',folder,'passed')
 for p in [folder/'status.json',folder.parent/'status.json']:
  d=json.loads(p.read_text());assert d['status']=='passed' and d['evaluation_receipt_sha256']==sha(folder/'evaluation_status.json')
 sync_model_status('grae_gpu',folder,'running')
 d=json.loads((folder/'status.json').read_text());assert d['status']=='partial' and d['official_evaluation']=='running' and 'evaluation_receipt_sha256' not in d

def test_stale_metrics_rejected(tmp_path):
 folder=fixture(tmp_path);(folder/'mini_val/official_eval/metrics_summary.json').write_text('{"amota":0.9}')
 with pytest.raises(ValueError,match='hash mismatch'):sync_model_status('grae_gpu',folder,'passed')
 assert json.loads((folder/'status.json').read_text())['status']=='partial'

def test_missing_split_rejected(tmp_path):
 folder=fixture(tmp_path);p=folder/'evaluation_status.json';d=json.loads(p.read_text());d['rows']=d['rows'][:1];p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='split coverage'):sync_model_status('grae_gpu',folder,'passed')
