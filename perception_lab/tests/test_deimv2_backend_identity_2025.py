import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import deimv2_backend_identity_2025 as identity

def setup_identity(tmp,monkeypatch):
 monkeypatch.setattr(identity,'ROOT',tmp)
 out=tmp/'reports/perception_2025_20261013/deimv2';out.mkdir(parents=True)
 for name,content in [('configs/perception_2025.yaml',json.dumps({'models':{'deimv2':{'source':'source','config':'cfg','checkpoint':'weight.pth','commit':'pin'}}})),('source/cfg','config'),('weight.pth','weights'),('weight.onnx','onnx'),('runs/perception_2025/deimv2/coco_predictions.json','[]')]:
  p=tmp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
 sha=identity.sha
 actual=dict(checkpoint_sha256=sha(tmp/'weight.pth'),model_config_sha256=sha(tmp/'source/cfg'),onnx_sha256=sha(tmp/'weight.onnx'))
 (out/'strict_load.json').write_text(json.dumps(actual))
 (out/'resource_manifest.json').write_text(json.dumps({'source_commit':'pin','files':[{'path':'cfg','sha256':actual['model_config_sha256']}]}))
 actual['source_manifest_sha256']=sha(out/'resource_manifest.json')
 (out/'export_identity.json').write_text(json.dumps(actual));(out/'status.json').write_text(json.dumps({'native_inference':'passed','prediction_sha256':sha(tmp/'runs/perception_2025/deimv2/coco_predictions.json')}))
 return tmp

@pytest.mark.parametrize('file',['weight.onnx','weight.pth','source/cfg','runs/perception_2025/deimv2/coco_predictions.json'])
def test_changed_backend_or_baseline_rejected(tmp_path,monkeypatch,file):
 setup_identity(tmp_path,monkeypatch);identity.verify_baseline(True)
 (tmp_path/file).write_text('changed')
 with pytest.raises(AssertionError):identity.verify_baseline(True)
