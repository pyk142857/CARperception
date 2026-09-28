import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
try:
 from failure_overlay import resolve_case
except ImportError:resolve_case=None
class FailureOverlayTests(unittest.TestCase):
 def test_fp_uses_original_index_fn_uses_gt_instance(self):
  self.assertIsNotNone(resolve_case)
  b=dict(center_xyz=[1,2,3],class_name='car')
  self.assertEqual(resolve_case(dict(kind='false_positive',module='detection',prediction_index=1,center_ego=[1,2,3],class_name='car'),[{},b],[],{}),b)
  self.assertEqual(resolve_case(dict(kind='false_negative',module='tracking',instance_token='g',center_ego=[1,2,3],class_name='car'),[],[],{'g':b}),b)
 def test_stale_case_is_rejected(self):
  self.assertIsNotNone(resolve_case)
  with self.assertRaises(ValueError):resolve_case(dict(kind='false_negative',module='detection',instance_token='g',center_ego=[9,2,3],class_name='car'),[],[],{'g':dict(center_xyz=[1,2,3],class_name='car')})
if __name__=='__main__':unittest.main()
