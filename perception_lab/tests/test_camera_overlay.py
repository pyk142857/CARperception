import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
try:
    from camera_overlay import project_segments
except ImportError:
    project_segments=None

class CameraOverlayTests(unittest.TestCase):
    def test_projection_uses_inverse_camera_pose(self):
        self.assertIsNotNone(project_segments)
        pose=np.eye(4);pose[0,3]=2
        k=np.array([[100,0,50],[0,100,50],[0,0,1]])
        lines=project_segments([[2,0,10],[3,0,10]],pose,k,100,100)
        np.testing.assert_allclose(lines,[[[50,50],[60,50]]])
    def test_segment_crossing_image_is_clipped(self):
        self.assertIsNotNone(project_segments)
        lines=project_segments([[-10,50,1],[110,50,1]],np.eye(4),np.eye(3),100,100)
        np.testing.assert_allclose(lines,[[[0,50],[99,50]]])
    def test_behind_camera_and_outside_image_are_rejected(self):
        self.assertIsNotNone(project_segments)
        for line in ([[0,0,-1],[1,1,-1]], [[200,200,1],[300,300,1]]):
            self.assertEqual(project_segments(line,np.eye(4),np.eye(3),100,100),[])
    def test_near_plane_crossing_stays_finite_and_inside(self):
        self.assertIsNotNone(project_segments)
        k=np.array([[100,0,50],[0,100,50],[0,0,1]])
        lines=project_segments([[1,0,-1],[0,0,2]],np.eye(4),k,100,100)
        self.assertEqual(len(lines),1)
        self.assertTrue(np.isfinite(lines).all())
        self.assertTrue((np.array(lines)>=0).all() and (np.array(lines)<=99).all())

if __name__=='__main__':unittest.main()
