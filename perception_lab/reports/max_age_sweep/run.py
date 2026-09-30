"""Controlled replay of fixed detections; only PubTracker.max_age changes."""
import sys,json,copy,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from pyquaternion import Quaternion
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'third_party/CenterPoint/tools/nusc_tracking')]
from pub_tracker import PubTracker,NUSCENES_TRACKING_NAMES
from evaluate_mini import load_frames,evaluate,write_csv
from rerun_mini import measured_results
packets=json.loads((ROOT/'manifests/frame_packets.json').read_text())
status=json.loads((ROOT/'results/status.json').read_text())
detections,detpath=measured_results(status,'M05','mini_scene',packets)
baseline,trackpath=measured_results(status,'M11','mini_tracking_M05',packets)
baseframes,_,_=load_frames(packets,detections,baseline)
basevalues=evaluate(baseframes,packets,.25,2.)[0]
inputs=[]
for rec,p in zip(detections,packets):
 T=np.array(p['sensors']['LIDAR_TOP']['T_ego_to_global']);q=Quaternion(matrix=T[:3,:3]);dets=[]
 for box in rec['boxes3d']:
  if box['class_name'] not in NUSCENES_TRACKING_NAMES:continue
  center=T[:3,:3]@np.array(box['center_xyz'])+T[:3,3];vel=T[:3,:3]@np.r_[box['velocity_xy'],0]
  dets.append(dict(translation=center.tolist(),size=box['size_wlh'],rotation=(q*Quaternion(box['rotation_wxyz'])).elements.tolist(),velocity=vel[:2].tolist(),detection_name=box['class_name'],detection_score=box['score']))
 inputs.append(dets)
summary=[];allcases={};checks={}
for age in [1,2,3,4,5,8,10,20]:
 tracker=PubTracker(hungarian=False,max_age=age);tracks=[];last_scene=None;last_time=None
 for rec,p,dets in zip(detections,packets,inputs):
  T=np.array(p['sensors']['LIDAR_TOP']['T_ego_to_global']);inv=np.linalg.inv(T);qinv=Quaternion(matrix=inv[:3,:3])
  if p['scene_token']!=last_scene:tracker.reset();last_time=p['reference_timestamp_us'];last_scene=p['scene_token']
  dt=(p['reference_timestamp_us']-last_time)/1e6;last_time=p['reference_timestamp_us']
  result=tracker.step_centertrack(copy.deepcopy(dets),dt);standard=[]
  for x in result:
   if x['active']==0:continue
   center=inv[:3,:3]@np.array(x['translation'])+inv[:3,3];vel=inv[:3,:3]@np.r_[x['velocity'],0]
   standard.append(dict(center_xyz=center.tolist(),size_wlh=x['size'],rotation_wxyz=(qinv*Quaternion(x['rotation'])).elements.tolist(),velocity_xy=vel[:2].tolist(),class_name=x['detection_name'],tracking_id=str(x['tracking_id']),score=x['detection_score']))
  tracks.append(dict(tracks3d=standard))
 if age==3:
  assert [t['tracks3d'] for t in tracks]==[t['tracks3d'] for t in baseline], 'Baseline replay mismatch'
  checks['baseline_predictions_exact_match']=True
 frames,_,_=load_frames(packets,detections,tracks)
 totals,rows,classes,cases,_=evaluate(frames,packets,.25,2.)
 if age==3:assert totals==basevalues;checks['baseline_metrics_exact_match']=True
 ev=Counter(c['kind'] for c in cases if c['module']=='tracking' and c['class_name']=='pedestrian')
 ped=next(c for c in classes if c['module']=='tracking' and c['class_name']=='pedestrian')
 row=dict(max_age=age,ped_id_switch=ev['id_switch'],ped_gap_id_change=ev['gap_id_change'],ped_gap_recovery=ev['gap_recovery'],ped_tp=ped['tp'],ped_fp=ped['fp'],ped_fn=ped['fn'],**{k:totals['tracking'][k] for k in ['tp','fp','fn']},**totals['tracking']['events'])
 summary.append(row);allcases[str(age)]=[c for c in cases if c['module']=='tracking' and c['kind'] in ['id_switch','gap_id_change','gap_recovery']]
 print(row,flush=True)
write_csv(HERE/'metrics.csv',summary)
(HERE/'identity_events.json').write_text(json.dumps(allcases,indent=2)+'\n')
files=[Path(detpath),Path(trackpath),ROOT/'manifests/frame_packets.json',ROOT/'tools/evaluate_mini.py',ROOT/'tools/evaluation_utils.py',ROOT/'third_party/CenterPoint/tools/nusc_tracking/pub_tracker.py',Path(__file__)]
(HERE/'verification.json').write_text(json.dumps(dict(checks=checks,scope='scene-0061 39 frames, score >= 0.25, evaluation gate < 2 m, greedy tracker, pedestrian association gate <= 1 m',sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
