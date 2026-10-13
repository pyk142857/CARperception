import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from perception_2025_visualize import load_artifacts, original_labels, ego_boxes, project_points, semantic_colors


def test_missing_artifacts_never_substitute(tmp_path):
    artifacts,harp,hashes=load_artifacts(tmp_path,'mini_val')
    assert artifacts=={'mambafusion':{},'grae_frozen_reference':{},'grae_mambafusion_native':{}}
    assert harp==hashes=={}
    p=tmp_path/'grae3dmot/frozen_reference/mini_val/tracking.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'results':{'one':[]}}))
    assert load_artifacts(tmp_path,'mini_val')[0]['grae_frozen_reference']=={}
    state={'status':'partial','input_family':'frozen_reference','inference':'passed','strict_load':'passed','official_evaluation':'passed'}
    (tmp_path/'grae3dmot/status.json').write_text(json.dumps(state))
    artifacts,_,hashes=load_artifacts(tmp_path,'mini_val')
    assert artifacts['grae_frozen_reference']=={'one':[]}
    assert artifacts['grae_mambafusion_native']=={}
    assert set(hashes)=={'grae_frozen_reference'}


def test_harp_full_original_coverage(tmp_path):
    p=tmp_path/'p.npz'
    np.savez(p,prediction=np.array([1,3,15],dtype=np.uint8),original_count=3,original_index=np.arange(3))
    assert original_labels({'prediction_path':str(p)},3).tolist()==[1,3,15]
    with pytest.raises(ValueError,match='alignment'):
        original_labels({'prediction_path':str(p)},4)
    np.savez(p,prediction=np.array([1,3,15],dtype=np.uint8),original_count=3,original_index=np.array([0,2,1]))
    with pytest.raises(ValueError,match='alignment'):
        original_labels({'prediction_path':str(p)},3)


def test_global_boxes_and_camera_projection():
    pose=np.eye(4);pose[:3,3]=[10,20,1]
    box={'translation':[11,22,4],'size':[2,4,2],'rotation':[1,0,0,0],'detection_score':.5,'detection_name':'car'}
    actual=ego_boxes([box],pose,'mambafusion')[0]
    assert actual['center_xyz']==[1,2,3]
    assert actual['track_id']=='mambafusion:car'
    assert ego_boxes([{**box,'detection_score':.1}],pose,'mambafusion')==[]
    camera=np.eye(4);camera[0,3]=1
    pixels,mask=project_points(np.array([[1,0,2],[1,0,-2],[101,0,2]]),camera,[[100,0,50],[0,100,50],[0,0,1]],100,100)
    assert mask.tolist()==[True,False,False]
    assert pixels.tolist()==[[50,50]]
    assert semantic_colors(np.array([0,255])).tolist()[1]==[55,55,55]


def test_stale_mamba_receipt_and_failed_state(tmp_path):
    from perception_2025_visualize import sha
    folder=tmp_path/'mambafusion';folder.mkdir()
    path=folder/'scene0061_predictions.json';path.write_text(json.dumps({'results':{'a':[]}}))
    status={'status':'partial','strict_load':'passed','inference':'passed','official_evaluation':'passed','official_metrics':{'scene0061':{'prediction_sha256':sha(path)}}}
    (folder/'status.json').write_text(json.dumps(status))
    assert load_artifacts(tmp_path,'scene0061')[0]['mambafusion']=={}
    (folder/'native_receipt.json').write_text(json.dumps({'predictions':{'scene0061':{'sha256':sha(path)}}}))
    assert load_artifacts(tmp_path,'scene0061')[0]['mambafusion']=={'a':[]}
    status['status']='running';(folder/'status.json').write_text(json.dumps(status))
    assert load_artifacts(tmp_path,'scene0061')[0]['mambafusion']=={}


def test_harp_partial_requires_explicit_native_completion(tmp_path):
    folder=tmp_path/'harpnext';folder.mkdir()
    status={'status':'partial','native_evaluation':'passed','strict_load':{'status':'passed'},'inference_frames':404}
    (folder/'status.json').write_text(json.dumps(status))
    (folder/'inference_progress.json').write_text(json.dumps({'status':'complete','inference_frames':404}))
    rows=[{'sample_token':str(i),'prediction_sha256':'receipt'} for i in range(404)]
    (folder/'inference_frames.json').write_text(json.dumps(rows))
    assert len(load_artifacts(tmp_path,'mini_val')[1])==404
    status['native_evaluation']='running';(folder/'status.json').write_text(json.dumps(status))
    assert load_artifacts(tmp_path,'mini_val')[1]=={}


def test_grae_native_shared_strict_receipt_and_input_hash(tmp_path):
    from perception_2025_visualize import sha
    m=tmp_path/'mambafusion';m.mkdir()
    prediction=m/'scene0061_predictions.json';prediction.write_text(json.dumps({'results':{'a':[]}}))
    digest=sha(prediction)
    (m/'status.json').write_text(json.dumps({'status':'partial','strict_load':'passed','inference':'passed','official_evaluation':'passed','official_metrics':{'scene0061':{'prediction_sha256':digest}}}))
    (m/'native_receipt.json').write_text(json.dumps({'predictions':{'scene0061':{'sha256':digest}}}))
    g=tmp_path/'grae3dmot/mambafusion_native';(g/'scene0061').mkdir(parents=True)
    tracking=g/'scene0061/tracking.json';tracking.write_text(json.dumps({'results':{'a':[]}}))
    (g/'status.json').write_text(json.dumps({'status':'partial','input_family':'mambafusion_native','inference':'passed','official_evaluation':'passed'}))
    manifest=g/'input_manifest.json';manifest.write_text(json.dumps({'splits':[{'split':'scene0061','sha256':digest}]}))
    (g.parent/'strict_load.json').write_text(json.dumps({'strict':True,'missing':[],'unexpected':[],'checkpoint_sha256':'checkpoint'}))
    receipt={'checkpoint_sha256':'checkpoint','tracking_sha256':{'scene0061':sha(tracking)},'input_manifest_sha256':sha(manifest)}
    (g/'native_receipt.json').write_text(json.dumps(receipt))
    assert load_artifacts(tmp_path,'scene0061')[0]['grae_mambafusion_native']=={'a':[]}
    manifest.write_text(json.dumps({'splits':[{'split':'scene0061','sha256':'stale'}]}))
    receipt['input_manifest_sha256']=sha(manifest)
    (g/'native_receipt.json').write_text(json.dumps(receipt))
    assert load_artifacts(tmp_path,'scene0061')[0]['grae_mambafusion_native']=={}
