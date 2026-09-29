import sys,tempfile,threading,unittest,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import start_case_browser as server

class ViewerAssetTests(unittest.TestCase):
 def test_extensionless_js_matches_pinned_vendor(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);old=server.RUNTIME;server.RUNTIME=root
   for directory,content in [('vendor',b'obsolete'),('vendor_'+server.VIEWER_VERSION,b'pinned')]:
    (root/directory).mkdir();(root/directory/'re_viewer.js').write_bytes(content)
   http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler);t=threading.Thread(target=http.serve_forever,daemon=True);t.start()
   try:
    base='http://127.0.0.1:'+str(http.server_port)
    for endpoint in ['/vendor/re_viewer','/vendor/re_viewer.js']:
     with urllib.request.urlopen(base+endpoint) as r:self.assertEqual(r.read(),b'pinned')
   finally:http.shutdown();http.server_close();t.join();server.RUNTIME=old
