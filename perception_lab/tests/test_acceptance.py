import unittest,pathlib,importlib.util
class AcceptanceTests(unittest.TestCase):
 def api(self):
  p=pathlib.Path(__file__).parents[1]/'tools/verify_artifacts.py'
  self.assertTrue(p.exists(),'Acceptance checker missing')
  s=importlib.util.spec_from_file_location('verify',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_paper_metric_cannot_count_as_measured(self):
  m=self.api();self.assertTrue(m.metric_errors({'source':'paper_reference','value':.9,'scope':'benchmark'}))
 def test_nan_and_missing_raw_evidence_rejected(self):
  m=self.api();self.assertTrue(m.metric_errors({'source':'measured','value':float('nan'),'scope':'benchmark'}))
 def test_demo_predictions_do_not_count_as_val_replay(self):
  m=self.api();self.assertTrue(m.coverage_errors([{'source':'measured','sample_token':'s','scope':'official_demo_not_nuscenes_smoke'}],['s']))
 def test_missing_token_rejected(self):
  m=self.api();self.assertTrue(m.coverage_errors([{'source':'measured','sample_token':'s1','scope':'replay'}],['s1','s2']))
if __name__=='__main__':unittest.main()
