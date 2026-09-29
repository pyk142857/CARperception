"""Serve the embedded Rerun case browser locally, with pinned vendor assets."""
import argparse,base64,hashlib,io,json,mimetypes,os,signal,subprocess,sys,tarfile,time,urllib.request,webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
from event_reviews import ReviewStore,Conflict
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'web/case_browser';RUNTIME=ROOT/'outputs/case_browser';STATE=RUNTIME/'server.json'
INTEGRITY='mXm12DzY+eFtvAc0PTh2ttDdU0xCopOMUsSumEdPJg2nSihJAhaoYypkxFyZHbGBSN6wu1tuymYuIehMCS5GEQ=='
VENDOR=['index.js','re_viewer.js','re_viewer_bg.wasm']

def setup():
 RUNTIME.mkdir(parents=True,exist_ok=True);vendor=RUNTIME/'vendor';vendor.mkdir(exist_ok=True)
 manifest=vendor/'sha256.json'
 if manifest.exists() and all((vendor/n).exists() and hashlib.sha256((vendor/n).read_bytes()).hexdigest()==json.loads(manifest.read_text()).get(n) for n in VENDOR):return
 data=urllib.request.urlopen('https://registry.npmjs.org/@rerun-io/web-viewer/-/web-viewer-0.23.4.tgz',timeout=90).read()
 if base64.b64encode(hashlib.sha512(data).digest()).decode()!=INTEGRITY:raise ValueError('Vendor integrity mismatch')
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  for n in VENDOR:(vendor/n).write_bytes(tar.extractfile('package/'+n).read())
 manifest.write_text(json.dumps({n:hashlib.sha256((vendor/n).read_bytes()).hexdigest() for n in VENDOR},indent=2))

def reviews():
 data=json.loads((ROOT/'reports/failure_events/events.json').read_text())
 targets={e['event_id'] for e in data['events']}|{g['group_id'] for g in data['groups']}
 return ReviewStore(RUNTIME/'reviews'/(data['dataset_id']+'.json'),data['dataset_id'],targets)

class Handler(BaseHTTPRequestHandler):
 def send_json(self,data,status=200,head=False):
  body=json.dumps(data,ensure_ascii=False).encode()
  self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
  self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers()
  if not head:self.wfile.write(body)
 def do_POST(self):
  if urlsplit(self.path).path!='/api/reviews':self.send_error(404);return
  origin=self.headers.get('Origin')
  if self.headers.get('Content-Type','').split(';')[0]!='application/json' or (origin and origin!='http://'+self.headers.get('Host','')):
   self.send_json({'error':'Unsupported request origin or content type'},403);return
  try:
   length=int(self.headers.get('Content-Length','0'))
   if not 0<length<=32000:raise ValueError('Request too large or empty')
   payload=json.loads(self.rfile.read(length))
   if not isinstance(payload,dict):raise ValueError('Expected object')
   self.send_json(reviews().save(payload))
  except Conflict as e:self.send_json({'error':str(e)},409)
  except (ValueError,KeyError) as e:self.send_json({'error':str(e)},400)

 def do_GET(self):self.serve(False)
 def do_HEAD(self):self.serve(True)
 def serve(self,head):
  path=urlsplit(self.path).path
  if path=='/api/reviews':
   try:self.send_json(reviews().read(),head=head)
   except Conflict as e:self.send_json({'error':str(e)},409,head=head)
   return
  routes={'/':APP/'index.html','/app.js':APP/'app.js','/style.css':APP/'style.css','/case_logic.mjs':APP/'case_logic.mjs',
          '/display_layers.mjs':APP/'display_layers.mjs','/event_ui.js':APP/'event_ui.js','/event_logic.mjs':APP/'event_logic.mjs',
          '/events.json':ROOT/'reports/failure_events/events.json',
          '/confidence.json':ROOT/'reports/confidence_filter/scores.json',
          '/cases.json':ROOT/'reports/mini_evaluation/cases.json','/summary.json':ROOT/'reports/mini_evaluation/summary.json',
          '/recording.json':ROOT/'outputs/rerun/triage_scene.json','/mini_scene.rrd':ROOT/'outputs/rerun/triage_scene.rrd',
          '/normal_targets.rrd':ROOT/'outputs/rerun/normal_targets.rrd'}
  routes.update({'/view_'+module+'_'+mode+'.rrd':ROOT/'outputs/rerun'/('view_'+module+'_'+mode+'.rrd')
                 for module in ['detection','tracking'] for mode in ['errors','normal']})
  routes.update({'/vendor/'+n:RUNTIME/'vendor'/n for n in VENDOR});routes['/vendor/re_viewer']=RUNTIME/'vendor/re_viewer.js'
  file=routes.get(path)
  if not file or not file.is_file():self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(file))[0] or 'application/octet-stream');self.send_header('Content-Length',str(file.stat().st_size));self.send_header('Cache-Control','no-cache');self.end_headers()
  if not head:
   try:
    with file.open('rb') as f:
     while chunk:=f.read(1024*1024):self.wfile.write(chunk)
   except (BrokenPipeError,ConnectionResetError):pass

def owned(pid):
 try:return 'start_case_browser.py' in Path(f'/proc/{pid}/cmdline').read_text().replace('\0',' ') and '--serve' in Path(f'/proc/{pid}/cmdline').read_text()
 except OSError:return False

def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=9092);p.add_argument('--serve',action='store_true');p.add_argument('--stop',action='store_true');p.add_argument('--no-browser',action='store_true');a=p.parse_args()
 if a.serve:ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever();return
 state=json.loads(STATE.read_text()) if STATE.exists() else {}
 if a.stop:
  if state.get('pid') and owned(state['pid']):os.kill(state['pid'],signal.SIGTERM)
  STATE.unlink(missing_ok=True);return
 if state.get('pid') and owned(state['pid']):url=state['url']
 else:
  if not (ROOT/'outputs/rerun/triage_scene.rrd').is_file():raise SystemExit('Export normal targets, then run rerun_mini.py --triage-only first')
  if not (ROOT/'reports/failure_events/events.json').is_file():raise SystemExit('Run tools/aggregate_failure_events.py in the mmdet3d environment first')
  setup();url=f'http://127.0.0.1:{a.port}/'
  with (RUNTIME/'server.log').open('a') as f:
   proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--serve','--port',str(a.port)],stdout=f,stderr=f,start_new_session=True,stdin=subprocess.DEVNULL)
  for _ in range(40):
   if proc.poll() is not None:raise RuntimeError('Server failed; check outputs/case_browser/server.log')
   try:
    urllib.request.urlopen(url,timeout=.5).close();break
   except OSError:time.sleep(.2)
  else:proc.terminate();raise RuntimeError('Server startup timeout')
  STATE.write_text(json.dumps(dict(pid=proc.pid,url=url),indent=2))
 print(url)
 if not a.no_browser:webbrowser.open(url)
if __name__=='__main__':main()
