"""Operational fixed-source recovery using reviewed receive/verify/install functions."""
import concurrent.futures,importlib.util,json,os,threading,time,urllib.request,subprocess
from pathlib import Path
spec=importlib.util.spec_from_file_location('recovery',Path('perception_lab/tools/recover_perception_weights_2025.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=m.SOURCES['mambafusion'];total=source['size'];blocksize=16*1024*1024
root=m.ROOT/'perception_lab/checkpoints/mambafusion_2025/recovery_staging';index=root/'blocks.json'
entries=m.verified_blocks(root,json.loads(index.read_text()) if index.exists() else [],total,blocksize)
stop=threading.Event();log_path=m.REPORT/'mamba_parallel_v2_attempts.json';logs=[]
proxy=os.environ.get('RECOVERY_PROXY','http://127.0.0.1:19099')
if not m.re.fullmatch(r'http://127[.]0[.]0[.]1:[0-9]+',proxy):raise ValueError('only task-local proxy endpoints allowed')
def worker(start):
 if stop.is_set():return None,[]
 end=min(start+blocksize,total)-1
 existing=next((e for e in entries if e['start']==start),None)
 if existing:return existing,[]
 part=root/f'{start:012d}.parallel.part';path=root/f'{start:012d}-{end:012d}.block'
 local=[]
 for attempt in range(2):
  event={'start':start,'end':end,'total':total,'attempt':attempt,'time_unix':time.time(),'route':proxy}
  try:
   opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy,'http':proxy}))
   request=urllib.request.Request(source['url'],headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
   with opener.open(request,timeout=45) as response:
    if response.status !=206:raise ValueError('parallel range requires 206; refusing whole-file duplication')
    m.receive(response,part,start,end,total,event)
   part.replace(path)
   entry={'start':start,'end':end,'size':end-start+1,'filename':path.name,'sha256':m.digest(path)}
   event['result']='verified_complete_block';local.append(event)
   m.save_json(root/f'{start:012d}.complete.json',entry)
   m.save_json(m.REPORT/'mamba_block_transport'/f'{start:012d}.json',local)
   return entry,local
  except Exception as exc:
   event.update(result='failed',error_type=type(exc).__name__)
   local.append(event)
   m.save_json(m.REPORT/'mamba_block_transport'/f'{start:012d}.json',local)
 stop.set()
 return None,local
starts=list(range(0,total,blocksize));begin=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(worker,start) for start in starts]
 for future in concurrent.futures.as_completed(futures):
  entry,events=future.result();logs.extend(events);m.save_json(log_path,logs)
  if entry is None:continue
  entries=[e for e in entries if e['start']!=entry['start']]+[entry]
  m.save_json(index,sorted(entries,key=lambda e:e['start']))
  print(json.dumps({'completed_blocks':len(entries),'total_blocks':len(starts),'verified_bytes':sum(e['size'] for e in entries),'seconds':round(time.time()-begin,2)}),flush=True)
if stop.is_set():raise RuntimeError('bounded transport failure; completed blocks preserved, switch route')
entries=m.verified_blocks(root,entries,total,blocksize)
if len(entries)!=len(starts):raise RuntimeError('missing valid blocks')
stage=root/'parallel-complete.staging'
with stage.open('wb') as out:
 for entry in sorted(entries,key=lambda e:e['start']):
  with (root/entry['filename']).open('rb') as inp:
   for data in iter(lambda:inp.read(1024*1024),b''):out.write(data)
 out.flush();os.fsync(out.fileno())
sha='d2454354b762a5a4b0b85af8b23c4b1f63da9357af7160aa313bc689a8d11ea6'
if stage.stat().st_size!=total or m.digest(stage)!=sha:raise RuntimeError('full Actions source hash mismatch')
subprocess.run(['python',str(m.ROOT/'perception_lab/tools/recover_perception_weights_2025.py'),'--model','mambafusion','--install-from',str(stage),'--relay-sha256',sha],check=True)
