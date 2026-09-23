import unittest,tempfile,pathlib
import numpy as np
from mmdet3d.datasets.transforms.loading import LoadPointsFromMultiSweeps
from mmdet3d.structures.points import LiDARPoints
class SweepLoaderTests(unittest.TestCase):
 def test_inverse_translation_is_rotated_into_target_frame(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'sweep.bin';np.array([[2,0,0,1,4]],dtype=np.float32).tofile(p)
   T=np.array([[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]],dtype=float)
   r={'points':LiDARPoints(np.zeros((1,5),np.float32),points_dim=5),'timestamp':2.,'lidar_sweeps':[{'timestamp':1.5,'lidar_points':{'lidar_path':str(p),'lidar2sensor':T}}]}
   actual=LoadPointsFromMultiSweeps(sweeps_num=1,use_dim=[0,1,2,3,4])(r)['points'].tensor.numpy()[1]
   expected=np.linalg.inv(T)@np.array([2,0,0,1])
   np.testing.assert_allclose(actual[:3],expected[:3],atol=1e-6)
   self.assertEqual(actual[4],.5)
if __name__=='__main__':unittest.main()
