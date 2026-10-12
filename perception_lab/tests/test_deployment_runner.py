"""Resume rejects altered artifacts; tracking needs its isolated dependencies."""
import importlib.util
import json
import sys
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('deployment_runner',Path(__file__).resolve().parents[1]/'tools/run_4090_deployment.py')
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)


def test_resume_reexecutes_after_artifact_corruption(tmp_path,monkeypatch):
    report=tmp_path/'report';output=report/'inputs/result.json';counter=tmp_path/'calls.txt'
    config=tmp_path/'config.yaml';config.write_text(json.dumps({'report':'report'}))
    script="from pathlib import Path; p=Path(%r); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{\"measured_fixture\":1}'); c=Path(%r); c.write_text(c.read_text()+'x' if c.exists() else 'x')"%(str(output),str(counter))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'commands',lambda *unused:[[sys.executable,'-c',script]])
    monkeypatch.setattr(runner,'fingerprint',lambda *unused:'same-input-and-config')
    monkeypatch.setattr(sys,'argv',['runner','--config',str(config),'--stage','preflight'])
    runner.main();assert counter.read_text()=='x'
    runner.main();assert counter.read_text()=='x'
    output.write_text('{"corrupt":true}')
    runner.main();assert counter.read_text()=='xx'
    state=json.loads((report/'runner_status.json').read_text())['preflight']
    assert state['artifacts'][str(output)]==runner.sha(output)


def test_tracking_evaluator_uses_environment_with_tracking_dependencies():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/deployment_4090.yaml').read_text())
    cmds=runner.commands(cfg,'sparse4d',root/'configs/deployment_4090.yaml')
    evaluation=[c for c in cmds if len(c)>1 and c[1]=='tools/sparse4d_evaluate.py']
    assert len(evaluation)==2
    assert all(c[0]==cfg['environments']['sparse4d'] for c in evaluation)
    cases=[c for c in cmds if len(c)>1 and c[1]=='tools/sparse4d_cases.py']
    assert cases[0][-2:]==['--out','outputs/sparse4d/scene0061']
