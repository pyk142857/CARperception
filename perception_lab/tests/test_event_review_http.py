import json,sys,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import start_case_browser as server

class ReviewHTTPTests(unittest.TestCase):
    def test_http_persistence_conflict_and_origin(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);report=root/'reports/failure_events';report.mkdir(parents=True)
            (report/'events.json').write_text(json.dumps(dict(dataset_id='test',
                events=[dict(event_id='event_a')],groups=[dict(group_id='group_a')])))
            oldroot,oldruntime=server.ROOT,server.RUNTIME
            server.ROOT=root;server.RUNTIME=root/'runtime'
            http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
            thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
            url='http://127.0.0.1:'+str(http.server_port)+'/api/reviews'
            payload=dict(dataset_id='test',target_id='event_a',revision=0,status='confirmed')
            def post(data,origin=None):
                headers={'Content-Type':'application/json'}
                if origin:headers['Origin']=origin
                return urllib.request.urlopen(urllib.request.Request(url,json.dumps(data).encode(),headers))
            try:
                with post(payload) as response:self.assertEqual(json.load(response)['revision'],1)
                with urllib.request.urlopen(url) as response:self.assertEqual(json.load(response)['items']['event_a']['status'],'confirmed')
                with self.assertRaises(urllib.error.HTTPError) as e:post(payload)
                self.assertEqual(e.exception.code,409)
                with self.assertRaises(urllib.error.HTTPError) as e:post(dict(payload,revision=1),'https://outside.invalid')
                self.assertEqual(e.exception.code,403)
                with self.assertRaises(urllib.error.HTTPError) as e:post(dict(payload,revision=1,status='resolved'))
                self.assertEqual(e.exception.code,400)
            finally:
                http.shutdown();http.server_close();thread.join()
                server.ROOT,server.RUNTIME=oldroot,oldruntime
