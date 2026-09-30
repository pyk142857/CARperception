import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from experimental_tracker import ExperimentalTracker,assign

def det(x,score=.9,cls='pedestrian',v=0):
 return dict(translation=[x,0,0],size=[1,1,2],rotation=[1,0,0,0],velocity=[v,0],detection_name=cls,detection_score=score)

class TrackerTests(unittest.TestCase):
 def test_global_assignment_resolves_greedy_conflict(self):
  cost=np.array([[.4,.6],[.5,1e18]])
  self.assertEqual(assign(cost,False).tolist(),[[0,0]])
  self.assertEqual(sorted(assign(cost,True).tolist()),[[0,1],[1,0]])
 def test_class_and_gate(self):
  t=ExperimentalTracker();t.step_centertrack([det(0)],0)
  self.assertEqual(t.step_centertrack([det(0,cls='car'),det(2)],.5)[0]['tracking_id'],2)
 def test_low_score_does_not_start_but_continues(self):
  t=ExperimentalTracker(two_stage=True,birth_score=.5)
  self.assertEqual(t.step_centertrack([det(0,.3)],0),[])
  ident=t.step_centertrack([det(0)],.5)[0]['tracking_id']
  self.assertEqual(t.step_centertrack([det(.2,.3)],.5)[0]['tracking_id'],ident)
 def test_expiry_and_empty_frames(self):
  t=ExperimentalTracker(motion='kalman',max_age=3);t.step_centertrack([det(0)],0)
  for expected in [1,1,0]:self.assertEqual(len(t.step_centertrack([],.5)),expected)
  self.assertEqual(t.step_centertrack([det(0)],.5)[0]['tracking_id'],2)
 def test_kalman_motion_and_reset(self):
  t=ExperimentalTracker(motion='kalman');t.step_centertrack([det(0,v=1)],0)
  t.step_centertrack([det(.5,v=1)],.5)
  self.assertTrue(np.isfinite(t.states[1][0]).all());self.assertGreater(t.states[1][0][0],0)
  t.reset();self.assertEqual(t.step_centertrack([det(4)],0)[0]['tracking_id'],1)

 def test_invalid_hungarian_pair_does_not_delete_old_track(self):
  t=ExperimentalTracker();t.step_centertrack([det(0)],0)
  result=t.step_centertrack([det(4)],.5)
  self.assertEqual({r['tracking_id']:r['active'] for r in result},{1:0,2:1})
