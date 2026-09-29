import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from triage_views import view_paths
class TriageViewsTests(unittest.TestCase):
 def test_errors_only_excludes_other_branch_and_normal(self):
  for module,other in [('detection','tracking'),('tracking','detection')]:
   bev,spatial,cameras=view_paths(module)
   paths=bev+spatial+[p for ps in cameras.values() for p in ps]
   self.assertFalse(any('/normal/' in p or '/'+other+'/' in p for p in paths))
   self.assertFalse(any(p.endswith('/**') and p in ['/ego/**','/**'] for p in paths))
 def test_normal_added_to_all_views(self):
  bev,spatial,cameras=view_paths('tracking',True)
  self.assertTrue(any('/normal/tracking/' in p for p in bev))
  self.assertTrue(any('/normal/tracking/' in p for p in spatial))
  self.assertTrue(all(any('/normal/tracking/' in p for p in paths) for paths in cameras.values()))
