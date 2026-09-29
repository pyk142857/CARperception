import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from selection_context import camera_eye

class ContextTests(unittest.TestCase):
 def test_camera_targets_box_without_modifying_points(self):
  item=dict(center=[20,-5,1],size=[1,1,2])
  position,target=camera_eye(item)
  self.assertEqual(target,[20,-5,1])
  self.assertGreater(np.linalg.norm(np.asarray(position)-target),5)
  self.assertGreater(position[2],target[2])
 def test_large_target_increases_camera_distance(self):
  small,_=camera_eye(dict(center=[0,0,0],size=[1,1,2]))
  large,_=camera_eye(dict(center=[0,0,0],size=[12,3,4]))
  self.assertGreater(np.linalg.norm(large),np.linalg.norm(small))
