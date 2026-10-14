import subprocess,time,json
from pathlib import Path
r=Path('/home/minglei/Desktop/CAR/perception_lab/reports/tracking_gpu_20261013/bev_pooling')
rows=[]
def snapshot(allowed=None):
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 foreign=[line for line in out.splitlines() if line.strip() and int(line.split(',')[0])!=allowed]
 rows.append(dict(time=time.time(),processes=out,external_compute=foreign));return foreign
if snapshot():raise RuntimeError('GPU occupied before pooling')
with (r/'benchmark.log').open('w') as f:
 p=subprocess.Popen(['bash','-c','source perception_lab/tools/bevfusion_trt_env.sh\nexec perception_lab/envs/optipmb_20261013/bin/python perception_lab/tools/bev_pooling_preindex_20261013.py benchmark --rounds 30 --warmup 10'],stdout=f,stderr=subprocess.STDOUT)
 while p.poll() is None:
  snapshot(p.pid);time.sleep(.5)
 snapshot(p.pid)
(r/'gpu_monitor.json').write_text(json.dumps(rows,indent=2)+'\n')
if any(x['external_compute'] for x in rows):
 if (r/'results.json').exists():(r/'results.json').rename(r/('rejected_results_'+str(int(time.time()))+'.json'))
 raise RuntimeError('Reject pooling due to competition')
if p.returncode:raise RuntimeError('Pooling failed with exit '+str(p.returncode))
print('Pooling completed; isolation samples',len(rows))
