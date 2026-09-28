import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
try:
    from maptr_geometry import lidar_to_image, vectors_to_ego, validate_frames
except ImportError:
    lidar_to_image = vectors_to_ego = validate_frames = None

class MapTRGeometryTests(unittest.TestCase):
    def test_projection_compensates_camera_capture_pose(self):
        self.assertIsNotNone(lidar_to_image, 'MapTR geometry is not implemented')
        reference=np.eye(4);reference[0,3]=10
        capture=np.eye(4);capture[0,3]=12
        lidar=dict(T_ego_to_global=reference,T_sensor_to_ego=np.eye(4))
        camera=dict(T_ego_to_global=capture,T_sensor_to_ego=np.eye(4),K=np.diag([100,100,1]))
        point=lidar_to_image(lidar,camera)@np.array([3,0,10,1])
        np.testing.assert_allclose(point[:2]/point[2],[10,0])
    def test_vector_axes_and_explicit_display_height(self):
        self.assertIsNotNone(vectors_to_ego, 'MapTR geometry is not implemented')
        extrinsic=np.eye(4);extrinsic[:2,:2]=[[0,1],[-1,0]];extrinsic[:3,3]=[1,2,1.8]
        lines=vectors_to_ego(np.array([[[0,10],[1,20]]]),extrinsic,ground_z=0)
        np.testing.assert_allclose(lines,[[[11,2,0],[21,1,0]]])
    def test_reject_corrupt_or_wrong_frame_predictions(self):
        self.assertIsNotNone(validate_frames, 'MapTR validation is not implemented')
        row=dict(sample_token='a',source='measured',status='passed',vectors=[dict(points_xyz=[[0,0,0],[1,1,0]],score=.8,class_name='divider')])
        validate_frames([row],[dict(sample_token='a')])
        with self.assertRaises(ValueError):validate_frames([row],[dict(sample_token='b')])
        row['vectors'][0]['points_xyz'][0][0]=float('nan')
        with self.assertRaises(ValueError):validate_frames([row],[dict(sample_token='a')])
    def test_empty_predictions_are_valid(self):
        self.assertIsNotNone(validate_frames, 'MapTR validation is not implemented')
        validate_frames([dict(sample_token='a',source='measured',status='passed',vectors=[])],[dict(sample_token='a')])

if __name__=='__main__':unittest.main()
