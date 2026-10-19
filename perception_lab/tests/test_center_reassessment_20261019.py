import copy,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from center_reassessment_20261019 import restore_native_center

def test_center_restoration_preserves_all_candidates_and_other_fields():
    sensor=np.eye(4);sensor[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    packets=[{'sample_token':'a','sensors':{'LIDAR_TOP':{'T_sensor_to_ego':sensor.tolist()}}}]
    boxes=[dict(center_xyz=[5,6,7],size_wlh=[2,3,4],rotation_wxyz=[1,0,0,0],score=s,class_name='car',velocity_xy=[1,2]) for s in [0,.1,.9]]
    records=[{'sample_token':'a','boxes3d':boxes,'source':'measured'}];old=copy.deepcopy(records)
    new=restore_native_center(records,packets)
    assert records==old and len(new[0]['boxes3d'])==3
    for original,fixed in zip(boxes,new[0]['boxes3d']):
        np.testing.assert_allclose(fixed['center_xyz'],[3,6,7])
        assert {k:v for k,v in original.items() if k!='center_xyz'}=={k:v for k,v in fixed.items() if k!='center_xyz'}

def test_backend_delta_changes_only_by_half_height_difference():
    packet=[{'sample_token':'a','sensors':{'LIDAR_TOP':{'T_sensor_to_ego':np.eye(4).tolist()}}}]
    source=[{'sample_token':'a','boxes3d':[{'center_xyz':[0,0,2],'size_wlh':[1,1,2]},{'center_xyz':[.1,0,2.3],'size_wlh':[1,1,2.4]}]}]
    fixed=restore_native_center(source,packet)[0]['boxes3d']
    np.testing.assert_allclose(np.array(fixed[1]['center_xyz'])-fixed[0]['center_xyz'],[.1,0,.1])
