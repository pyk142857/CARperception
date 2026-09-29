import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from triage_views import focus_bounds,focus_camera

class SelectionFocusTests(unittest.TestCase):
 def test_center_padding_and_small_target(self):
  x,y=focus_bounds([[10,20],[30,60]],minimum=100)
  self.assertEqual(sum(x)/2,20);self.assertEqual(sum(y)/2,40)
  self.assertGreaterEqual(x[1]-x[0],100)
  self.assertGreaterEqual(y[1]-y[0],100)
 def test_choose_largest_projection_and_missing_camera(self):
  cameras={'small':[[[0,0],[1,2]]],'large':[[[10,20],[30,60]]]}
  self.assertEqual(focus_camera(cameras),'large')
  self.assertIsNone(focus_camera({}))
 def test_edge_target_stays_centered(self):
  x,y=focus_bounds([[0,0],[4,8]],minimum=100)
  self.assertEqual(sum(x)/2,2);self.assertEqual(sum(y)/2,4)
