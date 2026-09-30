import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from experimental_tracker import ExperimentalTracker
from fast_candidate_tracker import FastCandidateTracker,clone_detection

def detection(x=0.,score=.9,cls='pedestrian'):
 return dict(translation=[x,0.,0.],size=[1.,1.,2.],rotation=[1.,0.,0.,0.],velocity=[0.,0.],detection_name=cls,detection_score=score)
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,list):return [plain(v) for v in x]
 return x
class FastTrackerTests(unittest.TestCase):
 def compare(self,frames):
  a=ExperimentalTracker(hungarian=False,two_stage=True,birth_score=.25);b=FastCandidateTracker()
  for ds,dt in frames:
   before=copy.deepcopy(ds)
   self.assertEqual(plain(a.step_centertrack(ds,dt)),plain(b.step_centertrack(ds,dt)))
   self.assertEqual(a.diagnostics,b.diagnostics);self.assertEqual(ds,before)
 def test_ties_thresholds_empty_and_expiry(self):
  self.compare([([detection(-.5),detection(.5)],0),([detection(0),detection(0,.3)],.5),([],1),([detection(0,.25),detection(4,.249)],.5),([detection(1.,.5)],.5),([detection(9,cls='car')],.5),([detection(9,cls='car')],.5),([detection(9,cls='car')],.5)])
 def test_random_multiframe_exact(self):
  rng=np.random.default_rng(42);frames=[]
  for _ in range(80):
   ds=[]
   for i in range(int(rng.integers(0,50))):
    d=detection(float(rng.uniform(-10,10)),float(rng.choice([.1,.25,.49,.5,.9])),str(rng.choice(['car','pedestrian','bicycle','barrier'])))
    d['translation'][1]=float(rng.uniform(-3,3));d['velocity']=rng.uniform(-2,2,2).tolist();ds.append(d)
   frames.append((ds,float(rng.choice([0.,.1,.5,1.]))))
  self.compare(frames)
 def test_output_does_not_alias_input(self):
  d=detection();b=FastCandidateTracker();out=b.step_centertrack([d],0)
  out[0]['translation'][0]=999;self.assertEqual(d['translation'][0],0.)
 def test_nested_extension_is_copied(self):
  d=detection();d['extra']={'nested':[1]};out=clone_detection(d);out['extra']['nested'][0]=2
  self.assertEqual(d['extra']['nested'],[1])
 def test_reset(self):
  b=FastCandidateTracker();b.step_centertrack([detection()],0);b.reset();self.assertEqual(b.step_centertrack([detection(10)],0)[0]['tracking_id'],1)
