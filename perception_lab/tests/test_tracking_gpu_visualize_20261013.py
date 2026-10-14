import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import tracking_gpu_visualize_20261013 as v


def build(tmp_path,monkeypatch):
    monkeypatch.setattr(v,'ROOT',tmp_path)
    def save(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return path
    raw=tmp_path/'raw.bin';np.zeros((2,5),dtype=np.float32).tofile(raw)
    packet=[{'sample_token':'one','sensors':{'LIDAR_TOP':{'path':str(raw),'T_sensor_to_ego':np.eye(4).tolist(),'T_ego_to_global':np.eye(4).tolist()}}}]
    packets=save(tmp_path/'packets.json',packet);det=save(tmp_path/'detections.json',{'results':{'one':[]}})
    entry={'split':'scene0061','frames':1,'packets_path':'packets.json','detection_path':'detections.json','packets_sha256':v.sha(packets),'detection_sha256':v.sha(det),'samples':[{'sample_token':'one','raw_lidar_sha256':v.sha(raw),'sensor_to_global':np.eye(4).tolist()}]}
    report=tmp_path/'report';save(report/'input_manifest.json',{'status':'passed','splits':[entry]})
    for model in ['grae','optipmb']:
        base=report/'native_compare'/model;folder=base/'frozen_reference' if model=='grae' else base
        pred=save(folder/'scene0061/tracking.json',{'results':{'one':[]}})
        metric=save(folder/'scene0061/official_eval/metrics_summary.json',{'amota':.5})
        save(folder/'scene0061/official_eval/protocol.json',{'prediction_sha256':v.sha(pred),'frozen_detection_sha256':v.sha(det),'packets_sha256':v.sha(packets),'tokens':['one']})
        save(folder/'evaluation_status.json',{'status':'passed','rows':[{'split':'scene0061','tracking_sha256':v.sha(pred),'metrics_sha256':v.sha(metric)}]})
        if model=='grae':
            save(folder/'status.json',{'status':'partial','inference':'passed','official_evaluation':'passed'})
            save(folder/'native_receipt.json',{'tracking_sha256':{'scene0061':v.sha(pred)}})
        else:save(base/'validation_summary.json',{'scene0061':{'tracking_file_sha256':v.sha(pred)}})
    return report


def test_verified_input_and_model_tokens(tmp_path,monkeypatch):
    report=build(tmp_path,monkeypatch)
    packets,artifacts,hashes=v.load_inputs('scene0061',report)
    assert len(packets)==1 and set(artifacts)=={'bev_detection','grae','optipmb'}
    assert set(hashes)>={'bev_detection','grae','optipmb','packets'}


def test_changed_frozen_source_fails(tmp_path,monkeypatch):
    report=build(tmp_path,monkeypatch)
    (tmp_path/'detections.json').write_text('{}')
    with pytest.raises(ValueError,match='Frozen source hash'):v.load_inputs('scene0061',report)


def test_stale_official_metrics_fails(tmp_path,monkeypatch):
    report=build(tmp_path,monkeypatch)
    path=report/'native_compare/grae/frozen_reference/scene0061/official_eval/metrics_summary.json'
    path.write_text('{"amota":0}')
    with pytest.raises(AssertionError):v.load_inputs('scene0061',report)
