"""Exercise serial orchestration only; no worker imports Torch or touches CUDA."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import deimv2_paired_onnx_2025 as paired


def test_controller_alternates_serial_requests_and_records_scope(tmp_path,monkeypatch):
    cfg=tmp_path/'configs/perception_2025.yaml';cfg.parent.mkdir()
    cfg.write_text(json.dumps({'environments':{'deimv2':'native_python','onnx':'onnx_python'}}))
    calls=[]
    class Process:
        def __init__(self,backend):
            self.backend=backend;self.stdin=self;self.stdout=self;self.ready=False;self.request={};self.done=False
        def write(self,line):
            self.request=json.loads(line)
            if 'index' in self.request:calls.append((self.backend,self.request['index']))
        def flush(self):pass
        def readline(self):
            if not self.ready:self.ready=True;return json.dumps({'ready':True})+'\n'
            i=self.request['index']
            return json.dumps({'finite':True,'input_sha256':str(i),'output_sha256':str(i),
                               'e2e_ms':2. if self.backend=='native' else 1.})+'\n'
        def poll(self):return 0 if self.done else None
        def wait(self,timeout):self.done=True;return 0
    import deimv2_backend_identity_2025 as identity
    monkeypatch.setattr(identity,'verify_baseline',lambda: {'scope':'unit fixture'})
    monkeypatch.setattr(paired,'ROOT',tmp_path)
    monkeypatch.setattr(paired,'OUT',tmp_path/'report/onnx')
    monkeypatch.setattr(paired.subprocess,'Popen',lambda cmd,**kwargs:Process(cmd[-1]))
    monkeypatch.setattr(sys,'argv',['paired'])
    paired.main()
    assert len(calls)==60
    for i in range(30):
        order=['native','onnx'] if i%2==0 else ['onnx','native']
        assert calls[2*i:2*i+2]==[(b,i%3) for b in order]
    result=json.loads((paired.OUT/'paired_benchmark.json').read_text())
    assert result['e2e_saving_ms']['mean']==1.
    assert all(result['same_backend_bitwise_repeat'].values())
    assert 'not isolated per-model' in result['memory_note']
