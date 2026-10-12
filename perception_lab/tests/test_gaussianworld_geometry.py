"""Check actual model adapter projection against fixed input contract."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from gaussianworld_4090 import prepare

def test_actual_mini_projection_and_preprocess():
    sys.path.insert(0,str(ROOT/'third_party/GaussianWorld'))
    from nuscenes.nuscenes import NuScenes
    nusc=NuScenes(version='v1.0-mini',dataroot=str(ROOT/'data/nuscenes'),verbose=False)
    packets=json.loads((ROOT/'reports/deployment_4090_20261008/inputs/scene0061_packets.json').read_text())
    for p in [packets[0],packets[19],packets[-1]]:
        image,meta,l2g,l2e,_=prepare(nusc,nusc.get('sample',p['sample_token']),None)
        assert image.shape==(1,6,3,864,1600)
        lidar=p['sensors']['LIDAR_TOP'];contract_l2g=np.array(lidar['T_ego_to_global'])@np.array(lidar['T_sensor_to_ego'])
        np.testing.assert_allclose(l2g,contract_l2g,atol=1e-10)
        np.testing.assert_allclose(l2e,lidar['T_sensor_to_ego'],atol=1e-10)
        for j,cam in enumerate(p['camera_order']):
            c=p['sensors'][cam];c2g=np.array(c['T_ego_to_global'])@np.array(c['T_sensor_to_ego']);K=np.eye(4);K[:3,:3]=c['K']
            np.testing.assert_allclose(meta[0]['lidar2img'][0][j],K@np.linalg.inv(c2g)@contract_l2g,atol=1e-9)
            # Original 900x1600 -> bottom crop 864x1600: v changes by -36.
            np.testing.assert_allclose(meta[0]['img_aug_matrix'][0,j,:2,3],[0,-36])
        point=np.array([2.,-3.,.5,1.]);np.testing.assert_allclose(np.linalg.inv(l2e)@(l2e@point),point,atol=1e-12)
