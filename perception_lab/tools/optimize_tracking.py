"""Sequential controlled tracker experiments; preserves deployed baseline."""
import sys,json,copy,hashlib,gzip,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from pyquaternion import Quaternion
from experimental_tracker import ExperimentalTracker
from evaluate_mini import load_frames,evaluate,write_csv
from rerun_mini import measured_results
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/tracking_optimization'
sys.path.insert(0,str(ROOT/'third_party/CenterPoint/tools/nusc_tracking'))
from pub_tracker import PubTracker,NUSCENES_TRACKING_NAMES
CONFIGS=[('S0_author',{'author':True,'hungarian':False}),('S0_mirror',{'hungarian':False}),
 ('S1_hungarian',{'author':True,'hungarian':True}),('S1_gated',{}),
 ('S2_kalman',{'motion':'kalman'}),('S3_rich_only',{'rich':True}),
 ('S3_kalman_rich',{'motion':'kalman','rich':True}),
 ('S4_greedy_high',{'hungarian':False,'two_stage':True,'birth_score':.5}),
 ('S4_greedy_low',{'hungarian':False,'two_stage':True,'birth_score':.25}),
 ('S4_high_birth',{'two_stage':True,'birth_score':.5}),
 ('S4_low_birth',{'two_stage':True,'birth_score':.25}),
 ('S4_kalman_rich_high',{'motion':'kalman','rich':True,'two_stage':True,'birth_score':.5}),
 ('S4_kalman_rich_low',{'motion':'kalman','rich':True,'two_stage':True,'birth_score':.25})]

def compressed(path,value):path.write_bytes(gzip.compress(json.dumps(value,separators=(',',':')).encode(),mtime=0))

def main():
 OUT.mkdir(exist_ok=True,parents=True)
 packets=json.loads((ROOT/'manifests/frame_packets.json').read_text());status=json.loads((ROOT/'results/status.json').read_text())
 detections,dp=measured_results(status,'M05','mini_scene',packets);baseline,tp=measured_results(status,'M11','mini_tracking_M05',packets)
 inputs=[]
 for rec,p in zip(detections,packets):
  T=np.array(p['sensors']['LIDAR_TOP']['T_ego_to_global']);q=Quaternion(matrix=T[:3,:3]);dets=[]
  for b in rec['boxes3d']:
   if b['class_name'] not in NUSCENES_TRACKING_NAMES:continue
   center=T[:3,:3]@np.array(b['center_xyz'])+T[:3,3];vel=T[:3,:3]@np.r_[b['velocity_xy'],0]
   dets.append(dict(translation=center.tolist(),size=b['size_wlh'],rotation=(q*Quaternion(b['rotation_wxyz'])).elements.tolist(),velocity=vel[:2].tolist(),detection_name=b['class_name'],detection_score=b['score']))
  inputs.append(dets)
 def run(cfg):
  cfg=dict(cfg);author=cfg.pop('author',False);tracker=(PubTracker if author else ExperimentalTracker)(max_age=3,**cfg)
  tracks=[];logs=[];times=[];last_scene=None;last_time=None
  for i,(p,dets) in enumerate(zip(packets,inputs)):
   T=np.array(p['sensors']['LIDAR_TOP']['T_ego_to_global']);inv=np.linalg.inv(T);qinv=Quaternion(matrix=inv[:3,:3])
   if p['scene_token']!=last_scene:tracker.reset();last_time=p['reference_timestamp_us'];last_scene=p['scene_token']
   dt=(p['reference_timestamp_us']-last_time)/1e6;last_time=p['reference_timestamp_us']
   start=time.perf_counter();result=tracker.step_centertrack(copy.deepcopy(dets),dt);times.append((time.perf_counter()-start)*1000)
   standard=[]
   for x in result:
    if x['active']==0:continue
    center=inv[:3,:3]@np.array(x['translation'])+inv[:3,3];vel=inv[:3,:3]@np.r_[x['velocity'],0]
    standard.append(dict(center_xyz=center.tolist(),size_wlh=x['size'],rotation_wxyz=(qinv*Quaternion(x['rotation'])).elements.tolist(),velocity_xy=vel[:2].tolist(),class_name=x['detection_name'],tracking_id=str(x['tracking_id']),score=x['detection_score']))
   tracks.append(dict(tracks3d=standard));logs.append(dict(frame=i,associations=getattr(tracker,'diagnostics',[])))
  return tracks,logs,times
 summary=[];allclasses=[];allframes=[];saved={};checks={};admission_keys=['id_switch','gap_id_change','ped_id_switch','ped_gap_id_change','fp','fn','ped_fp','ped_fn','front_fn','mixed_observations']
 for name,cfg in CONFIGS:
  tracks,logs,times=run(cfg);repeated,_,_=run(cfg);assert tracks==repeated;checks[name+'_repeat']=True
  if name=='S0_author':assert [t['tracks3d'] for t in tracks]==[t['tracks3d'] for t in baseline];checks['original_baseline_exact']=True
  if name=='S0_mirror':assert tracks==saved['S0_author'];checks[name+'_author_exact']=True
  saved[name]=tracks
  frames,_,_=load_frames(packets,detections,tracks);totals,fr,classes,cases,details=evaluate(frames,packets,.25,2.)
  ids=defaultdict(Counter)
  for (module,i),(gt,pred,matched,fn,fp) in details.items():
   if module=='tracking':
    for g,p,d in matched:ids[pred[p]['tracking_id']][gt[g]['instance_token']]+=1
  mixed=sum(sum(c.values())-max(c.values()) for c in ids.values());mixed_tracks=sum(len(c)>1 for c in ids.values())
  ev=Counter(c['kind'] for c in cases if c['module']=='tracking' and c['class_name']=='pedestrian')
  ped=next(c for c in classes if c['module']=='tracking' and c['class_name']=='pedestrian')
  frontfn=sum(c['module']=='tracking' and c['kind']=='false_negative' and 0<=c['center_ego'][0]<=30 and abs(c['center_ego'][1])<=3 for c in cases)
  row=dict(variant=name,**{k:totals['tracking'][k] for k in ['tp','fp','fn','mean_center_error_m']},**{k:totals['tracking']['events'].get(k,0) for k in ['id_switch','gap_id_change','gap_recovery']},ped_id_switch=ev['id_switch'],ped_gap_id_change=ev['gap_id_change'],ped_fp=ped['fp'],ped_fn=ped['fn'],front_fn=frontfn,mixed_tracks=mixed_tracks,mixed_observations=mixed,association_ms_mean=float(np.mean(times)),association_ms_p95=float(np.percentile(times,95)))
  base=summary[0] if summary else row
  row['candidate_pass']=(row['id_switch']<base['id_switch'] and row['ped_id_switch']<base['ped_id_switch'] and row['id_switch']+row['gap_id_change']<=base['id_switch']+base['gap_id_change'] and row['ped_id_switch']+row['ped_gap_id_change']<=base['ped_id_switch']+base['ped_gap_id_change'] and all(row[k]<=base[k] for k in ['fp','fn','ped_fp','ped_fn','front_fn','mixed_observations']))
  summary.append(row);allclasses.extend(dict(variant=name,**r) for r in classes if r['module']=='tracking');allframes.extend(dict(variant=name,**r) for r in fr if r['module']=='tracking')
  compressed(OUT/(name+'_tracks.json.gz'),tracks);compressed(OUT/(name+'_associations.json.gz'),logs)
  (OUT/(name+'_identity.json')).write_text(json.dumps([c for c in cases if c['module']=='tracking' and c['kind'] in ['id_switch','gap_id_change']],indent=2)+'\n')
  print(json.dumps(row),flush=True)
 write_csv(OUT/'metrics.csv',summary);write_csv(OUT/'classes.csv',allclasses);write_csv(OUT/'frames.csv',allframes)
 (OUT/'configs.json').write_text(json.dumps(dict(CONFIGS),indent=2)+'\n')
 paths=[Path(dp),Path(tp),ROOT/'manifests/frame_packets.json',Path(__file__),ROOT/'tools/experimental_tracker.py',ROOT/'tools/evaluate_mini.py',ROOT/'tools/evaluation_utils.py',ROOT/'third_party/CenterPoint/tools/nusc_tracking/pub_tracker.py']
 (OUT/'verification.json').write_text(json.dumps(dict(checks=checks,frames=len(packets),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},input_score_range=[min(d['detection_score'] for f in inputs for d in f),max(d['detection_score'] for f in inputs for d in f)],note='timing includes diagnostic collection; no GT input to trackers; original-label fixed-threshold diagnostic; no deployment'),indent=2)+'\n')
if __name__=='__main__':main()
