"""Run in OptiPMB venv: python -m unittest discover -s perception_lab/tests -p test_optipmb_tracking_20261013.py"""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
P=Path(__file__).resolve().parents[1]/'tools/optipmb_tracking_20261013.py'
spec=importlib.util.spec_from_file_location('optipmb_adapter',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Contracts(unittest.TestCase):
    def cloud(self):
        return {'position':np.array([[0.,3.,3.01,10.],[0.,0.,0.,0.],[0.,0.,0.,0.]]),'position_global':np.array([[0.,3.,3.01,10.],[0.,0.,0.,0.],[0.,0.,0.,0.]]),'remission':np.array([1.,2.,3.,4.])}
    def box(self,name='car',score=.5):
        return {'translation':[3.5,0.,0.],'size':[2.,4.,6.],'rotation':[1.,0.,0.,0.],'velocity':[4.,5.],'detection_name':name,'detection_score':score}
    def test_mapping_and_allowlist(self):
        z=m.measurement([self.box(),self.box('barrier')],self.cloud())
        np.testing.assert_array_equal(z['bbox_hwlxyza'],[[6.,2.,4.,3.5,0.,0.,0.,0.,0.]])
        np.testing.assert_array_equal(z['bbox_velocity'],[[4.,5.]])
        np.testing.assert_array_equal(z['bbox_quat'],[[1.,0.,0.,0.]])
        self.assertEqual(z['bbox_class'].tolist(),['car'])
        np.testing.assert_array_equal(z['pts_meas'][3],self.cloud()['remission'])
    def test_real_author_distance_point_count_and_strict_score(self):
        model=m.gen_OptiPMB_filter_model(m.json.loads(m.CONFIG.read_text()),m.CLASSES)
        tracker=m.native.OptiPMB_Filter(model)
        p=tracker.preprocess_measurments(m.measurement([self.box()],self.cloud()))
        self.assertEqual(p['lidar_meas'].shape[1],2) # exactly 3m removed
        self.assertEqual(p['used_bbox_pt_num'].tolist(),[1])
        p=tracker.preprocess_measurments(m.measurement([self.box(score=model['car']['keep_score_thld'])],self.cloud()))
        self.assertEqual(len(p['used_bbox_score']),0) # author uses strict >
    def test_online_export_uses_last_mean_and_author_quaternion(self):
        est={'id':[17], 'mean':[np.array([[1,2],[3,4],[5,6],[0,0.5]])], 'aux_info':[[0,0,'car',22,7,0,.3]],'hwla':[[8,9,10,1,0,0,0]]}
        labels=[];r=m.online_boxes(est,'token',labels)[0]
        self.assertEqual(r['translation'],[2,4,7]);self.assertEqual(r['size'],[9,10,8]);self.assertEqual(r['rotation'],[1,0,0,0]);self.assertEqual(r['tracking_id'],'car_0')
        self.assertEqual(m.online_boxes(est,'other',labels)[0]['tracking_id'],'car_0')
    def test_native_fastmurty_known_optimum(self): self.assertEqual(m.known_murty()['cost'].tolist(),[-17.])

if __name__=='__main__': unittest.main()
