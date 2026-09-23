import unittest,importlib.util,pathlib
class GeometryTests(unittest.TestCase):
 def test_cross_time_transform_uses_both_ego_poses(self):
  path=pathlib.Path(__file__).parents[1]/'tools/geometry.py'
  self.assertTrue(path.exists(),'Geometry implementation absent')
  s=importlib.util.spec_from_file_location('geometry',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  import numpy as np
  source=np.eye(4);source[0,3]=10
  target=np.eye(4);target[0,3]=13
  sensor=np.eye(4);sensor[0,3]=1
  T=m.sensor_to_sensor(source,sensor,target,np.eye(4))
  np.testing.assert_allclose((T@np.array([5,0,0,1]))[:3],[3,0,0])
  np.testing.assert_allclose(np.linalg.inv(T)@T,np.eye(4),atol=1e-12)
if __name__=='__main__':unittest.main()
