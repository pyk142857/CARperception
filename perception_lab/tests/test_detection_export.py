import sys,unittest,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
try:
 from export_nuscenes_detection import convert_box, export_records
except ImportError:
 convert_box=export_records=None

class ExportTests(unittest.TestCase):
 def box(self):return dict(center_xyz=[1,2,3],size_wlh=[2,4,1.5],rotation_wxyz=[1,0,0,0],velocity_xy=[2,0],class_name='car',score=.15)
 def test_pose_and_no_extra_center_shift(self):
  self.assertIsNotNone(convert_box)
  b=convert_box(self.box(),'t',dict(translation=[10,20,0],rotation=[2**-.5,0,0,2**-.5]))
  np.testing.assert_allclose(b['translation'],[8,21,3]);np.testing.assert_allclose(b['velocity'],[0,2],atol=1e-12)
  self.assertEqual(b['size'],[2,4,1.5]);self.assertEqual(b['attribute_name'],'');self.assertEqual(b['detection_score'],.15)
 def test_reject_missing_velocity_and_invalid_box(self):
  self.assertIsNotNone(convert_box)
  for key,value in [('velocity_xy',None),('score',float('nan')),('size_wlh',[0,1,2]),('rotation_wxyz',[2,0,0,0]),('class_name','unknown')]:
   b=self.box();b[key]=value
   with self.assertRaises(ValueError):convert_box(b,'t',dict(translation=[0,0,0],rotation=[1,0,0,0]))
 def test_full_coverage_empty_frame_and_stable_top500(self):
  self.assertIsNotNone(export_records)
  records=[dict(sample_token='t',source='measured',boxes3d=[self.box() for _ in range(501)]),dict(sample_token='empty',source='measured',boxes3d=[])]
  poses={t:dict(translation=[0,0,0],rotation=[1,0,0,0]) for t in ['t','empty']}
  result,audit=export_records(records,poses,['t','empty'])
  self.assertEqual(len(result['results']['t']),500);self.assertEqual(result['results']['empty'],[]);self.assertEqual(audit['removed_top500'],1)
  with self.assertRaises(ValueError):export_records(records,poses,['t'])
  with self.assertRaises(ValueError):export_records(records+records[:1],poses,['t','empty'])
if __name__=='__main__':unittest.main()
