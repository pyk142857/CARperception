"""Audit upstream asynchronous camera projection and native→ego→global boxes."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
lab=Path(__file__).resolve().parents[1]
nusc=NuScenes(version='v1.0-mini',dataroot=str(lab/'data/nuscenes'),verbose=False)
infos=pickle.load(open(a.out/'selected_infos.pkl','rb'))['infos']
native=json.loads((a.out/'native_predictions.json').read_text());records=json.loads((a.out/'predictions.json').read_text())
def mat(rec):
    m=np.eye(4);m[:3,:3]=Quaternion(rec['rotation']).rotation_matrix;m[:3,3]=rec['translation'];return m
checks=[]
for ix in sorted(set([0,len(infos)//2,len(infos)-1])):
    info=infos[ix];sample=nusc.get('sample',info['token']);lid=nusc.get('sample_data',sample['data']['LIDAR_TOP'])
    l2e=mat(nusc.get('calibrated_sensor',lid['calibrated_sensor_token']));e2g=mat(nusc.get('ego_pose',lid['ego_pose_token']))
    projections=[]
    for cam,c in info['cams'].items():
        sd=nusc.get('sample_data',sample['data'][cam]);cal=nusc.get('calibrated_sensor',sd['calibrated_sensor_token']);pose=nusc.get('ego_pose',sd['ego_pose_token'])
        independent=np.linalg.inv(mat(cal))@np.linalg.inv(mat(pose))@e2g@l2e
        author=np.eye(4);author[:3,:3]=np.linalg.inv(c['sensor2lidar_rotation']);author[:3,3]=-author[:3,:3]@c['sensor2lidar_translation']
        error=float(np.max(np.abs(author-independent)))
        assert error<1e-8
        projections.append(dict(camera=cam,sensor_to_lidar_inverse_max_abs=error,camera_lidar_timestamp_delta_us=sd['timestamp']-lid['timestamp']))
    err=[]
    for x,b in zip(native[ix]['boxes_3d'],records[ix]['boxes3d']):
        c=(np.linalg.inv(l2e)@np.r_[b['center_xyz'],1])[:3]
        err.append(float(np.max(np.abs(c-np.array(x[:3])))))
        np.testing.assert_allclose(b['size_wlh'],np.array(x)[[4,3,5]],atol=1e-7)
    checks.append(dict(sample_token=info['token'],frame_index=ix,cameras=projections,native_ego_roundtrip_center_max_abs=max(err)))
(a.out/'input_coordinate_audit.json').write_text(json.dumps(dict(frames=checks,native_center='gravity',native_frame='LIDAR_TOP',standard_frame='LIDAR_TOP timestamp ego',size_conversion='native[4,3,5] → wlh',gt_used_for_prediction=False),indent=2)+'\n')
print(json.dumps(checks,indent=2))
