import importlib.util
from pathlib import Path
import unittest
import numpy as np
P=Path(__file__).parents[1]/'tools/grae3dmot_2025.py'
class GraeTests(unittest.TestCase):
 def test_each_family_retains_status_and_metrics(self):
  import tempfile,json
  from unittest.mock import patch
  m=self.module()
  with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)):
   old={'input':'frozen_reference','split':'mini_val','frames':81,'amota':.8}
   new={'input':'mambafusion_native','split':'mini_val','frames':81,'amota':.9}
   state={'status':'partial','inference':'passed','input_family':'frozen_reference','official_evaluation':'passed'}
   m.write(m.OUT/'frozen_reference/status.json',state);m.write(m.OUT/'metrics.json',[old])
   m.write(m.OUT/'mambafusion_native/status.json',dict(state,input_family='mambafusion_native',official_evaluation='pending'))
   for split in ['scene0061','mini_val']:m.write(m.OUT/'mambafusion_native'/split/'tracking.json',{})
   m.write(m.OUT/'mambafusion_native/native_receipt.json',{'tracking_sha256':{split:m.sha(m.OUT/'mambafusion_native'/split/'tracking.json') for split in ['scene0061','mini_val']}})
   with patch.object(m,'evaluate',return_value=[new]):m.postprocess({},'mambafusion_native','evaluate')
   self.assertEqual(json.loads((m.OUT/'metrics.json').read_text()),[old,new])
   self.assertEqual(json.loads((m.OUT/'frozen_reference/status.json').read_text()),state)
   with patch.object(m,'evaluate',side_effect=RuntimeError('new failure')):
    with self.assertRaises(RuntimeError):m.postprocess({},'mambafusion_native','evaluate')
   self.assertEqual(json.loads((m.OUT/'frozen_reference/status.json').read_text()),state)
 def test_evaluation_receipt_and_metrics_require_successful_call(self):
  import tempfile,json
  from unittest.mock import patch
  m=self.module()
  with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)):
   native={'status':'partial','inference':'passed','input_family':'frozen_reference','official_evaluation':'pending'}
   m.write(m.OUT/'status.json',native)
   rows=[{'input':'frozen_reference','split':'mini_val','frames':81,'amota':.5}]
   with patch.object(m,'evaluate',return_value=rows) as evaluate:
    self.assertEqual(m.postprocess({},'frozen_reference','evaluate'),0);evaluate.assert_called_once()
   self.assertEqual(json.loads((m.OUT/'metrics.json').read_text()),rows)
   self.assertEqual(json.loads((m.OUT/'evaluate_status.json').read_text())['status'],'passed')
   with patch.object(m,'evaluate',side_effect=RuntimeError('failed evaluator')):
    with self.assertRaises(RuntimeError):m.postprocess({},'frozen_reference','evaluate')
   self.assertEqual(json.loads((m.OUT/'evaluate_status.json').read_text())['status'],'failed')
   self.assertEqual(json.loads((m.OUT/'status.json').read_text())['official_evaluation'],'failed')
 def test_optimization_resource_failure_preserves_native_status(self):
  import tempfile,json,sys
  from unittest.mock import patch
  m=self.module()
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);cfg=folder/'config.json';cfg.write_text('{}')
   original={'status':'partial','inference':'passed','official_evaluation':'passed'}
   m.write(folder/'status.json',original)
   with patch.object(m,'OUT',folder),patch.object(m,'SOURCE',folder/'absent_source'),patch.object(m,'environment',return_value={}),patch.object(m,'prepare',side_effect=FileNotFoundError('missing input')),patch.object(sys,'argv',['grae','--phase','optimize','--config',str(cfg)]):
    self.assertEqual(m.main(),2)
   self.assertEqual(json.loads((folder/'status.json').read_text()),original)
   self.assertEqual(json.loads((folder/'optimization.json').read_text())['status'],'blocked_resource')
 def test_mamba_default_paths_match_detector_outputs(self):
  m=self.module()
  paths=m.sources({},'mambafusion_native')
  self.assertEqual(paths['mini_val'],m.ROOT/'reports/perception_2025_20261013/mambafusion/mini_val_predictions.json')
 def module(self):
  self.assertTrue(P.exists(),'GRAE mini adapter missing')
  spec=importlib.util.spec_from_file_location('grae_adapter',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_gt_fields_cannot_enter_reader(self):
  m=self.module();b={'translation':[1,2,3],'size':[2,4,1.5],'rotation':[1,0,0,0],'velocity':[1,0],'detection_score':.9,'detection_name':'car','tracking_id':'GROUND_TRUTH','future_box':[0]}
  x=m.frame_arrays([b],{'sample_token':'x','scene_token':'s','timestamp_us':1500000},nms=False)
  self.assertEqual(x['tracking_id'].tolist(),[-1]);self.assertNotIn('future_box',x);self.assertEqual(x['timestamp'],1.5)
 def test_empty_shapes_and_seven_classes(self):
  m=self.module();x=m.frame_arrays([],{'sample_token':'x','scene_token':'s','timestamp_us':0},nms=False)
  self.assertEqual(x['translation'].shape,(0,3));self.assertEqual(x['rotation'].shape,(0,4));self.assertEqual(len(m.CLASSES),7)
 def test_official_module_rename_preserves_all_tensor_objects(self):
  m=self.module();x=np.array([1,2]);result=m.checkpoint_name_adapter({'spatial_mfl.gamma.weight':x})
  self.assertIs(result['spatial_dffl.gamma.weight'],x)
  with self.assertRaises(ValueError):m.checkpoint_name_adapter({'spatial_mfl.gamma.weight':x,'spatial_dffl.gamma.weight':x})
 def test_strict_prefix_normalization_rejects_mixed(self):
  m=self.module()
  with self.assertRaises(ValueError):m.normalize_weights({'module.a':np.array(1),'b':np.array(2)})
  self.assertEqual(list(m.normalize_weights({'module.a':np.array(1)})),['a'])
 def test_export_timer_preserves_original_loop_values(self):
  import types,sys
  from unittest.mock import patch
  class Toy:
   def _test_mini_seq(self,values):
    dets=values;draw_track=list(reversed(values));out=[]
    for idx in range(len(dets)):
     out.append(dets[idx]*2)
    for idx in range(len(draw_track)):
     out.append(draw_track[idx]+1)
    return out
  original=types.ModuleType('trainer.predictor');original.BasePredictor=Toy
  package=types.ModuleType('trainer');package.predictor=original
  with patch.dict(sys.modules,{'trainer':package,'trainer.predictor':original}):
   timed=self.module().instrument_export_loops(Toy())
   self.assertEqual(timed._test_mini_seq([1,3,7]),Toy()._test_mini_seq([1,3,7]))
   self.assertGreaterEqual(timed._profile_export_ms,0.)
if __name__=='__main__':unittest.main()
