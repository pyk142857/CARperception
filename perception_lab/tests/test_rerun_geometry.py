import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from rerun_mini import box_components, camera_to_reference, transform_points

class RerunGeometryTests(unittest.TestCase):
    def test_nuscenes_box_axes_and_quaternion(self):
        c,s,q=box_components([dict(center_xyz=[1,2,3],size_wlh=[2,4,1],rotation_wxyz=[1,0,0,0])])
        np.testing.assert_equal(s,[[4,2,1]])
        np.testing.assert_equal(q,[[0,0,0,1]])
        np.testing.assert_equal(c,[[1,2,3]])
    def test_camera_uses_capture_pose(self):
        reference=np.eye(4);reference[0,3]=10
        capture=np.eye(4);capture[0,3]=12
        extrinsic=np.eye(4);extrinsic[1,3]=1
        t=camera_to_reference(dict(T_ego_to_global=capture,T_sensor_to_ego=extrinsic),reference)
        np.testing.assert_allclose(transform_points([[0,0,0]],t),[[2,1,0]])
    def test_world_trail_to_current_ego(self):
        pose=np.eye(4);pose[:2,:2]=[[0,-1],[1,0]];pose[0,3]=10
        p=[[1,0,0],[2,0,0]]
        np.testing.assert_allclose(transform_points(transform_points(p,pose),np.linalg.inv(pose)),p)
    def test_empty_and_invalid_boxes(self):
        self.assertEqual(box_components([])[0].shape,(0,3))
        with self.assertRaises(ValueError):
            box_components([dict(center_xyz=[0,0,0],size_wlh=[-1,1,1],rotation_wxyz=[1,0,0,0])])

if __name__=='__main__':unittest.main()
