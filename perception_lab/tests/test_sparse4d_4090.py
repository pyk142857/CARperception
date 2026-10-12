"""Coordinate and identity contract regression with nontrivial extrinsic rotation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import torch
from pyquaternion import Quaternion
from sparse4d_4090 import native_to_ego
from export_nuscenes_detection import convert_box


def test_native_lidar_to_ego_and_global_roundtrip():
    q=Quaternion(axis=[0,0,1],degrees=90)
    native=dict(boxes_3d=torch.tensor([[2.,3.,4.,8.,2.,1.5,.3,5.,6.,0.]]),scores_3d=torch.tensor([.8]),labels_3d=torch.tensor([0]),instance_ids=torch.tensor([7]),cls_scores=torch.tensor([.9]))
    info=dict(lidar2ego_rotation=q.elements.tolist(),lidar2ego_translation=[1,2,3])
    box=native_to_ego(native,info,['car'],'scene_a')[0]
    np.testing.assert_allclose(box['center_xyz'],[-2,4,7],atol=1e-6)
    np.testing.assert_allclose(box['size_wlh'],[2,8,1.5])
    np.testing.assert_allclose(box['velocity_xy'],[-6,5],atol=1e-6)
    assert box['track_id']=='scene_a:7'
    assert native_to_ego(native,info,['car'],'scene_b')[0]['track_id']!='scene_a:7'
    global_box=convert_box(box,'sample',dict(rotation=q.inverse.elements.tolist(),translation=[-2,1,-3]))
    np.testing.assert_allclose(global_box['translation'],[2,3,4],atol=1e-6)
    np.testing.assert_allclose(global_box['velocity'],[5,6],atol=1e-6)
    np.testing.assert_allclose(Quaternion(global_box['rotation']).rotation_matrix,Quaternion(axis=[0,0,1],radians=.3).rotation_matrix,atol=1e-6)
