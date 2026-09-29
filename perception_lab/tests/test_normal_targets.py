import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from export_normal_targets import normal_indices

class NormalTargetsTests(unittest.TestCase):
 def test_excludes_localization_and_identity_failures(self):
  pred=[dict(prediction_index=i,tracking_id=str(i)) for i in range(5)]
  matched=[(i,i,.2) for i in range(4)]
  cases=[dict(kind='center_error_over_1m',matched_prediction_index=1),
         dict(kind='id_switch',tracking_id='2'),
         dict(kind='false_positive',prediction_index=4),
         dict(kind='gap_recovery',tracking_id='3')]
  self.assertEqual(normal_indices(pred,matched,cases),[0,3])
 def test_empty_or_unmatched_not_normal(self):
  self.assertEqual(normal_indices([dict(prediction_index=0)],[],[]),[])
