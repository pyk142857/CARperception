"""Reject backend comparisons after weights, source, config or baseline change."""
import json
from deimv2_2025 import ROOT,sha

def verify_baseline(check_predictions=False):
 out=ROOT/'reports/perception_2025_20261013/deimv2'
 c=json.loads((ROOT/'configs/perception_2025.yaml').read_text())['models']['deimv2'];source=ROOT/c['source'];native=json.loads((out/'strict_load.json').read_text());manifest=json.loads((out/'resource_manifest.json').read_text());receipt=json.loads((out/'export_identity.json').read_text())
 if 'checkpoint_sha256' not in receipt:receipt=json.loads((out/'backend_identity_verification.json').read_text())
 actual=dict(checkpoint_sha256=sha(ROOT/c['checkpoint']),model_config_sha256=sha(source/c['config']),onnx_sha256=sha((ROOT/c['checkpoint']).with_suffix('.onnx')),source_manifest_sha256=sha(out/'resource_manifest.json'))
 assert actual['checkpoint_sha256']==native['checkpoint_sha256'],'Native checkpoint changed'
 assert actual['model_config_sha256']==native['model_config_sha256'],'Native config changed'
 assert all(actual[k]==receipt[k] for k in actual),'Export identity does not match current backend inputs'
 assert manifest['source_commit']==c['commit']
 for item in manifest['files']:assert sha(source/item['path'])==item['sha256'],item['path']
 if check_predictions:
  status=json.loads((out/'status.json').read_text());assert status.get('native_inference')=='passed'
  assert sha(ROOT/'runs/perception_2025/deimv2/coco_predictions.json')==status['prediction_sha256'],'Baseline predictions changed'
 return dict(**actual,provenance_scope=receipt.get('scope','export receipt'))
