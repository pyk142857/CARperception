import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from event_reviews import ReviewStore,Conflict

class ReviewTests(unittest.TestCase):
    def test_persist_conflict_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'reviews.json'
            store=ReviewStore(path,'v1',{'event_a','group_b'})
            item=dict(dataset_id='v1',target_id='event_a',revision=0,status='confirmed',owner='tester',root_cause='',action='',validation='',duplicate_of='')
            saved=store.save(item)
            self.assertEqual(saved['revision'],1)
            self.assertEqual(ReviewStore(path,'v1',{'event_a','group_b'}).read()['items']['event_a']['status'],'confirmed')
            with self.assertRaises(Conflict):store.save(item)
            with self.assertRaises(ValueError):store.save(dict(item,revision=1,status='resolved'))
            with self.assertRaises(ValueError):store.save(dict(item,revision=1,status='duplicate',duplicate_of='event_a'))
            with self.assertRaises(Conflict):ReviewStore(path,'v2',{'event_a'}).read()
            store.save(dict(item,revision=1,status='resolved',validation='Regression checked'))
            self.assertEqual(len(store.read()['history']),2)
