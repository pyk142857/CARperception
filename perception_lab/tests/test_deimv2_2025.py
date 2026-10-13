import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'tools/deimv2_2025.py'
s=importlib.util.spec_from_file_location('deim2025',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_sparse_ids_and_xywh_no_score_filter():
    assert m.coco_rows(19,[1],[[10,20,30,55]],[.001],[1,3,7])==[{'image_id':19,'category_id':3,'bbox':[10.,20.,20.,35.],'score':.001}]
def test_invalid_geometry_and_nonfinite_rejected():
    with pytest.raises(ValueError):m.coco_rows(1,[0],[[10,20,9,55]],[.3],[1])
    with pytest.raises(ValueError):m.coco_rows(1,[0],[[0,0,1,1]],[float('nan')],[1])
def test_unknown_category_rejected():
    with pytest.raises(ValueError):m.coco_rows(1,[2],[[0,0,1,1]],[.3],[1,3])

def test_complete_native_manifest_requires_and_protects_predictions(tmp_path):
    import json
    import sys
    sys.path.insert(0,str(p.parent))
    import run_perception_2025 as runner
    out=tmp_path/'out';run=tmp_path/'run';out.mkdir();run.mkdir()
    m.write(out/'status.json',{'status':'passed','coco_images':5000})
    with pytest.raises(FileNotFoundError):m.publish_native_manifest(out,run)
    assert not (out/'artifact_manifest.json').exists()
    for name in ['coco_predictions.json.gz','mini_camera_predictions.json.gz','image_ids.json',
                 'official_eval/metrics.json','official_eval/protocol.json','author_faster_coco_eval/metrics.json',
                 'profile.json','timings.json','strict_load.json','hooks.json','deploy_equivalence.json',
                 'three_image_gate.json','resolved_config.json','resource_manifest.json']:
        m.write(out/name,{})
    m.write(run/'coco_predictions.json',[]);m.write(run/'mini_camera_predictions.json',[])
    m.publish_native_manifest(out,run)
    manifest=out/'artifact_manifest.json'
    assert runner.output_proof(manifest,out/'status.json') is not None
    (run/'mini_camera_predictions.json').write_text('[1]')
    assert runner.output_proof(manifest,out/'status.json') is None
