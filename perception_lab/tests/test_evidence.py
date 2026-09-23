import importlib.util,unittest,tempfile,pathlib
SPEC=importlib.util.spec_from_file_location('evidence',pathlib.Path(__file__).parents[1]/'tools/evidence.py')
class EvidenceTests(unittest.TestCase):
 def api(self):
  self.assertTrue(pathlib.Path(SPEC.origin).exists(),'Evidence validator not implemented')
  m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m);return m
 def test_changed_input_cannot_resume(self):
  m=self.api()
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'input';p.write_text('first');a=m.hash_files([p]);p.write_text('changed');self.assertNotEqual(a,m.hash_files([p]))
 def test_corrupt_artifact_cannot_resume(self):
  m=self.api()
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'output';p.write_text('real result');r={'status':'passed','fingerprint':'x','artifacts':m.hash_files([p])}
   self.assertTrue(m.reusable(r,'x'));p.write_text('corrupt');self.assertFalse(m.reusable(r,'x'))
 def test_empty_artifacts_and_failed_record_rejected(self):
  m=self.api();self.assertFalse(m.reusable({'status':'passed','fingerprint':'x','artifacts':{}},'x'));self.assertFalse(m.reusable({'status':'failed'},'x'))
if __name__=='__main__':unittest.main()
