import unittest,subprocess,tempfile,pathlib,json,sys
ROOT=pathlib.Path(__file__).parents[1]
class PipelineTests(unittest.TestCase):
 def test_failure_does_not_stop_independent_jobs_or_pass(self):
  self.assertTrue((ROOT/'tools/run_pipeline.py').exists(),'Pipeline not implemented')
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);plan=root/'plan.json'
   plan.write_text(json.dumps({'root':d,'modules':{'M00':{'jobs':{'smoke':{'command':[sys.executable,'-c','raise SystemExit(7)']}}},'M01':{'jobs':{'smoke':{'command':[sys.executable,'-c',"from pathlib import Path;Path('result.json').write_text('{}')"],'artifacts':['result.json']}}}}}))
   r=subprocess.run([sys.executable,str(ROOT/'tools/run_pipeline.py'),'--plan',str(plan),'--phase','baseline'],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0)
   status=json.loads((root/'results/status.json').read_text())
   self.assertEqual(status['modules']['M00']['smoke']['status'],'failed')
   self.assertEqual(status['modules']['M01']['smoke']['status'],'passed')
if __name__=='__main__':unittest.main()
