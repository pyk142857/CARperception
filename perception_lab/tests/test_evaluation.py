import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
try:
    from evaluation_utils import match_boxes, identity_events
except ImportError:
    match_boxes = None

def box(x, cls='car', score=1, ident='a'):
    return dict(translation=[x,0,0],class_name=cls,score=score,instance_token=ident,tracking_id=ident)

class EvaluationTests(unittest.TestCase):
    def test_duplicate_and_wrong_class_are_false_positives(self):
        self.assertIsNotNone(match_boxes)
        m,fn,fp=match_boxes([box(0)], [box(.2,score=.5),box(.1,score=.9),box(0,'truck')],2)
        self.assertEqual([(g,p) for g,p,d in m],[(0,1)])
        self.assertEqual(fn,[]);self.assertEqual(fp,[0,2])
    def test_distance_gate_strict_and_empty_inputs(self):
        self.assertIsNotNone(match_boxes)
        self.assertEqual(match_boxes([box(0)],[box(2)],2),([], [0], [0]))
        self.assertEqual(match_boxes([],[],2),([],[],[]))
    def test_tracking_assignment_and_continuity(self):
        self.assertIsNotNone(match_boxes)
        gt=[box(0,ident='g1'),box(1.5,ident='g2')]
        pred=[box(.6,ident='p1'),box(-1,ident='p2')]
        m,fn,fp=match_boxes(gt,pred,2,tracking=True)
        self.assertEqual(len(m),2)
        # Existing valid association should survive nearest-neighbour crossing.
        m,_,_=match_boxes([box(0,ident='g1')],[box(.1,ident='new'),box(.5,ident='old')],2,
                          tracking=True,previous={'g1':'old'})
        self.assertEqual(m[0][1],1)
    def test_switch_gap_and_scene_separation(self):
        self.assertIsNotNone(match_boxes)
        history={}
        self.assertEqual(identity_events(history,'s1',0,[('g','p1')]),[])
        e=identity_events(history,'s1',1,[('g','p2')])
        self.assertEqual(e[0]['kind'],'id_switch')
        e=identity_events(history,'s1',4,[('g','p3')])
        self.assertEqual(e[0]['kind'],'gap_id_change')
        self.assertEqual(e[0]['missed_frames'],2)
        self.assertEqual(identity_events(history,'s2',5,[('g','p4')]),[])
        e=identity_events(history,'s1',6,[('g','p3')])
        self.assertEqual(e[0]['kind'],'gap_recovery')

if __name__=='__main__':unittest.main()
