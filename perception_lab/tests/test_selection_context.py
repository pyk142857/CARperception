import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from selection_context import context_mask,context_extent

class ContextTests(unittest.TestCase):
 def test_crop_tracks_target_in_ego_coordinates(self):
  item=dict(center=[20,-5,1],size=[1,1,2])
  points=np.array([[20,-5,1],[22,-5,1],[0,0,0],[40,-5,1]])
  self.assertEqual(context_mask(points,item).tolist(),[True,True,False,False])
  self.assertEqual(context_extent(item),3)
  self.assertGreater(context_extent(dict(size=[12,3,4])),9)
 def test_empty_cloud(self):
  self.assertEqual(len(context_mask(np.empty((0,3)),dict(center=[0,0,0],size=[1,1,2]))),0)
