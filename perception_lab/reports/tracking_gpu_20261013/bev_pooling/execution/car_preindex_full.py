import subprocess,time,json,os
from pathlib import Path
r=Path('perception_lab/reports/tracking_gpu_20261013/bev_pooling');rows=[]
def sample(pid=None):
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
 foreign=[x for x in raw.splitlines() if x.strip() and int(x.split(',')[0])!=pid]
 rows.append(dict(time=time.time(),allowed_pid=pid,processes=raw,external_compute=foreign));return foreign
for split in ['scene0061','mini_val']:
 for mode in ['upstream','preindex']:
  if sample():raise RuntimeError('GPU not free')
  out=r/'full_model'/mode/split
  cmd=['bash','/tmp/car_preindex_run.sh','--packets',f'perception_lab/reports/deployment_4090_20261008/inputs/{split}_packets.json','--out',str(out),'--capture-pooling',f'perception_lab/runs/tracking_gpu_20261013/bev_pooling/full_model_captures/{mode}/{split}']
  if mode=='preindex':cmd+=['--paired-pooling-rounds','30']
  with (r/f'full_{mode}_{split}.log').open('w') as f:
   p=subprocess.Popen(cmd,env=dict(os.environ,CAR_BEV_POOLING=mode),stdout=f,stderr=subprocess.STDOUT)
   while p.poll() is None:sample(p.pid);time.sleep(1)
  sample(p.pid);(r/'full_model_gpu_monitor.json').write_text(json.dumps(rows,indent=2)+'\n')
  if p.returncode:raise RuntimeError(f'{mode} {split} exit {p.returncode}')
  if any(x['external_compute'] for x in rows):raise RuntimeError('Contested measurements; reject timings')
  print(mode,split,'passed',flush=True)
