import unittest,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from case_labels import LabelStore
from event_reviews import Conflict

class LabelTests(unittest.TestCase):
 def test_persistence_history_conflicts_and_fp_semantics(self):
  with tempfile.TemporaryDirectory() as d:
   cases=[dict(case_id='fp',kind='false_positive',frame=2),dict(case_id='fn',kind='false_negative',frame=2)]
   store=LabelStore(Path(d)/'labels.json','v',cases)
   payload=dict(dataset_id='v',case_id='fp',revision=0,label='gt_missing')
   self.assertEqual(store.save(payload)['label'],'gt_missing')
   self.assertEqual(store.read()['items']['fp']['original']['kind'],'false_positive')
   with self.assertRaises(Conflict):store.save(payload)
   with self.assertRaises(ValueError):store.save(dict(payload,case_id='fn'))
   with self.assertRaises(ValueError):store.save(dict(payload,case_id='unknown'))
   with self.assertRaises(ValueError):store.save(dict(payload,label=[]))
   with self.assertRaises(Conflict):store.save(dict(payload,dataset_id='wrong',revision=1))
   store.save(dict(payload,label='original',revision=1))
   self.assertEqual(store.read()['items']['fp']['label'],'original')
   self.assertEqual(len(store.read()['history']),2)
