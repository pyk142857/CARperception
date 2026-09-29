import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_failure_confidence import case_confidence,score_bin
class ConfidenceTests(unittest.TestCase):
 def test_fn_has_no_score_and_id_uses_new_track(self):
  self.assertIsNone(case_confidence(dict(kind='false_negative'),[]))
  case=dict(kind='id_switch',tracking_id='new',class_name='car')
  boxes=[dict(tracking_id='old',class_name='car',score=.9),dict(tracking_id='new',class_name='car',score=.4)]
  self.assertEqual(case_confidence(case,boxes),.4)
  with self.assertRaises(ValueError):case_confidence(case,[])
 def test_bins_edges(self):
  self.assertEqual(score_bin(.25),'[0.25,0.40)')
  self.assertEqual(score_bin(.4),'[0.40,0.60)')
  self.assertEqual(score_bin(1.),'[0.80,1.00]')
  self.assertEqual(score_bin(None),'N/A')
