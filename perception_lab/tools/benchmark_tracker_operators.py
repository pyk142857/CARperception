"""Paired CPU operator benchmark and exact differential validation on mini inputs."""
import sys,json,gzip,hashlib,time,cProfile,pstats,platform,os,csv
from pathlib import Path
import numpy as np
import scipy
from pyquaternion import Quaternion
from experimental_tracker import ExperimentalTracker,NAMES
from fast_candidate_tracker import FastCandidateTracker
from rerun_mini import measured_results
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/tracking_operator_optimization'

def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,list):return [plain(v) for v in x]
 return x

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 packets=json.loads((ROOT/'manifests/frame_packets.json').read_text());status=json.loads((ROOT/'results/status.json').read_text())
 records,dp=measured_results(status,'M05','mini_scene',packets)
 inputs=[];last=None;last_scene=None
 for rec,p in zip(records,packets):
  T=np.array(p['sensors']['LIDAR_TOP']['T_ego_to_global']);q=Quaternion(matrix=T[:3,:3]);dets=[]
  for b in rec['boxes3d']:
   if b['class_name'] not in NAMES:continue
   center=T[:3,:3]@np.array(b['center_xyz'])+T[:3,3];vel=T[:3,:3]@np.r_[b['velocity_xy'],0]
   dets.append(dict(translation=center.tolist(),size=b['size_wlh'],rotation=(q*Quaternion(b['rotation_wxyz'])).elements.tolist(),velocity=vel[:2].tolist(),detection_name=b['class_name'],detection_score=b['score']))
  ts=p['reference_timestamp_us'];reset=last_scene!=p['scene_token'];dt=0 if reset else (ts-last)/1e6;last=ts;last_scene=p['scene_token'];inputs.append((dets,dt,reset))
 factories={'reference':lambda:ExperimentalTracker(hungarian=False,two_stage=True,birth_score=.25),'optimized':FastCandidateTracker}
 def run(kind,capture=False):
  t=factories[kind]();durations=[];snapshots=[];logs=[]
  for ds,dt,reset in inputs:
   if reset:t.reset()
   start=time.perf_counter_ns();out=t.step_centertrack(ds,dt);durations.append((time.perf_counter_ns()-start)/1e6)
   if capture:snapshots.append(plain(out));logs.append(plain(t.diagnostics))
  return durations,snapshots,logs
 original_inputs=json.dumps(inputs,sort_keys=True)
 _,a,la=run('reference',True);_,b,lb=run('optimized',True)
 assert a==b,'Full output mismatch';assert la==lb,'Diagnostic mismatch';assert json.dumps(inputs,sort_keys=True)==original_inputs,'Input mutated'
 # Compare against the preceding experiment's archived, ego-frame active outputs.
 canonical=[]
 for p,frame in zip(packets,b):
  T=np.array(p['sensors']['LIDAR_TOP']['T_ego_to_global']);inv=np.linalg.inv(T);q=Quaternion(matrix=inv[:3,:3]);standard=[]
  for x in frame:
   if not x['active']:continue
   center=inv[:3,:3]@np.array(x['translation'])+inv[:3,3];vel=inv[:3,:3]@np.r_[x['velocity'],0]
   standard.append(dict(center_xyz=center.tolist(),size_wlh=x['size'],rotation_wxyz=(q*Quaternion(x['rotation'])).elements.tolist(),velocity_xy=vel[:2].tolist(),class_name=x['detection_name'],tracking_id=str(x['tracking_id']),score=x['detection_score']))
  canonical.append(dict(tracks3d=standard))
 archived=ROOT/'reports/tracking_optimization/S4_greedy_low_tracks.json.gz'
 assert canonical==json.loads(gzip.decompress(archived.read_bytes())), 'Prior candidate mismatch'
 for kind in factories:
  for _ in range(3):run(kind)
 samples=[];rounds=[]
 for rep in range(30):
  for kind in (['reference','optimized'] if rep%2==0 else ['optimized','reference']):
   ds,_,_=run(kind)
   samples.extend(dict(round=rep,variant=kind,frame=i,ms=v) for i,v in enumerate(ds))
   rounds.append(dict(round=rep,variant=kind,mean_ms=float(np.mean(ds))))
 for kind in factories:
  prof=cProfile.Profile();prof.runcall(run,kind)
  with (OUT/('profile_'+kind+'.txt')).open('w') as f:pstats.Stats(prof,stream=f).strip_dirs().sort_stats('cumtime').print_stats(30)
 summaries={}
 for kind in factories:
  vals=[r['ms'] for r in samples if r['variant']==kind];means=[r['mean_ms'] for r in rounds if r['variant']==kind]
  summaries[kind]=dict(mean_ms=float(np.mean(vals)),p50_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),round_mean_min_ms=min(means),round_mean_max_ms=max(means),samples=len(vals))
 speedup=summaries['reference']['mean_ms']/summaries['optimized']['mean_ms']
 paths=[Path(dp),ROOT/'manifests/frame_packets.json',archived,ROOT/'tools/experimental_tracker.py',ROOT/'tools/fast_candidate_tracker.py',Path(__file__)]
 result=dict(checks=dict(all_39_frames_exact=True,diagnostics_exact=True,input_unchanged=True,archived_candidate_exact=True),performance=summaries,speedup=speedup,latency_reduction_percent=(1-1/speedup)*100,rounds=30,warmup_rounds=3,frames=len(inputs),input_boxes=sum(len(i[0]) for i in inputs),environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),''),logical_cpus=os.cpu_count()),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 for name,rs in [('samples.csv',samples),('rounds.csv',rounds)]:
  with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
