"""Author CenterPoint tracker on measured mini detections; no GT identities."""
import argparse,json,sys,time,copy
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
import cv2
from evidence import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--detections',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(root/'third_party/CenterPoint/tools/nusc_tracking'))
 from pub_tracker import PubTracker,NUSCENES_TRACKING_NAMES
 records=json.loads(Path(a.detections).read_text());packets=json.loads((root/'manifests/frame_packets.json').read_text())
 assert [r['sample_token'] for r in records]==[p['sample_token'] for p in packets]
 assert all(r['source']=='measured' and r['status']=='passed' for r in records)
 def run():
  tracker=PubTracker(hungarian=False,max_age=3);last_scene=None;last_time=None;outputs=[];durations=[]
  for rec,packet in zip(records,packets):
   T=np.array(packet['sensors']['LIDAR_TOP']['T_ego_to_global']);q=Quaternion(matrix=T[:3,:3]);dets=[]
   for box in rec['boxes3d']:
    if box['class_name'] not in NUSCENES_TRACKING_NAMES:continue
    if box['velocity_xy'] is None:raise ValueError('Measured velocity required')
    center=T[:3,:3]@np.array(box['center_xyz'])+T[:3,3];vel=T[:3,:3]@np.r_[box['velocity_xy'],0]
    dets.append(dict(translation=center.tolist(),size=box['size_wlh'],rotation=(q*Quaternion(box['rotation_wxyz'])).elements.tolist(),velocity=vel[:2].tolist(),detection_name=box['class_name'],detection_score=box['score']))
   if packet['scene_token']!=last_scene:tracker.reset();last_time=packet['reference_timestamp_us'];last_scene=packet['scene_token']
   dt=(packet['reference_timestamp_us']-last_time)/1e6;last_time=packet['reference_timestamp_us'];start=time.perf_counter()
   tracks=tracker.step_centertrack(copy.deepcopy(dets),dt);durations.append((time.perf_counter()-start)*1000)
   frame=[]
   for x in tracks:
    if x['active']==0:continue
    frame.append(dict(sample_token=rec['sample_token'],translation=x['translation'],size=x['size'],rotation=x['rotation'],velocity=x['velocity'],tracking_id=str(x['tracking_id']),tracking_name=x['detection_name'],tracking_score=x['detection_score']))
   outputs.append(frame)
  return outputs,durations
 tracks,times=run();repeated,_=run()
 if tracks!=repeated:raise RuntimeError('Scene reset / independent-run reproducibility failed')
 native={'meta':{'use_camera':False,'use_lidar':True,'use_radar':False,'use_map':False,'use_external':False},'results':{r['sample_token']:t for r,t in zip(records,tracks)}}
 (out/'native_tracking.json').write_text(json.dumps(native,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from nuscenes.utils.data_classes import LidarPointCloud
 history={};images=[];bundles=[]
 for i,(rec,packet,frame) in enumerate(zip(records,packets,tracks)):
  T=np.array(packet['sensors']['LIDAR_TOP']['T_ego_to_global']);inv=np.linalg.inv(T);qinv=Quaternion(matrix=inv[:3,:3]);standard=[]
  points=LidarPointCloud.from_file(packet['sensors']['LIDAR_TOP']['path']).points[:3];lidar_ego=np.array(packet['sensors']['LIDAR_TOP']['T_sensor_to_ego']);points=lidar_ego[:3,:3]@points+lidar_ego[:3,3:4]
  fig,ax=plt.subplots(figsize=(8,8));ax.scatter(points[0,::5],points[1,::5],s=.15,c='gray')
  for x in frame:
   ident=x['tracking_id'];center=inv[:3,:3]@np.array(x['translation'])+inv[:3,3];velocity=inv[:3,:3]@np.r_[x['velocity'],0]
   history.setdefault(ident,[]).append(x['translation']);trail=np.array(history[ident][-20:]);trail=trail@inv[:3,:3].T+inv[:3,3]
   color=plt.get_cmap('tab20')(int(ident)%20);ax.plot(trail[:,0],trail[:,1],color=color,linewidth=1);ax.scatter(center[0],center[1],color=color,s=8);ax.text(center[0],center[1],ident,fontsize=6,color=color)
   standard.append(dict(center_xyz=center.tolist(),size_wlh=x['size'],rotation_wxyz=(qinv*Quaternion(x['rotation'])).elements.tolist(),velocity_xy=velocity[:2].tolist(),class_name=x['tracking_name'],tracking_id=ident,score=x['tracking_score']))
  ax.set(xlim=(-55,55),ylim=(-55,55),xlabel='ego x (m)',ylabel='ego y (m)',title=f"MINI CenterPoint → author tracker | {i+1}/39\nscene-0061 | {packet['timestamp_us']} us");ax.set_aspect('equal')
  image=out/f'{i:06d}_tracks.png';fig.savefig(image,dpi=110);plt.close(fig);images.append(image)
  bundles.append(dict(run_id=out.relative_to(root/'runs').parts[0],module_id='M11',model_id='CenterPoint_PubTracker',source='measured',scope='mini_scene_engineering_only',status='passed',sample_token=rec['sample_token'],scene_token=rec['scene_token'],timestamp_us=rec['timestamp_us'],native_coordinate_system='global centre xyz wlh quaternion',label_namespace='nuScenes_tracking_7',checkpoint_sha256=rec['checkpoint_sha256'],source_config_hash=sha256(root/'third_party/CenterPoint/tools/nusc_tracking/pub_tracker.py'),tracks3d=standard))
 image=cv2.imread(str(images[0]));h,w=image.shape[:2];writer=cv2.VideoWriter(str(out/'mini_tracking_M05.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(w,h))
 if not writer.isOpened():raise RuntimeError('Video writer unavailable')
 for path in images:writer.write(cv2.imread(str(path)))
 writer.release()
 (out/'predictions.json').write_text(json.dumps(bundles,indent=2))
 (out/'frame_index.json').write_text(json.dumps([dict(sample_token=p['sample_token'],scene_token=p['scene_token'],timestamp_us=p['timestamp_us'],video_fps=2,source_interval_seconds=0 if i==0 else (p['timestamp_us']-packets[i-1]['timestamp_us'])/1e6) for i,p in enumerate(packets)],indent=2))
 (out/'summary.json').write_text(json.dumps({'source':'measured','scope':'mini_scene_engineering_only','sample_count':len(records),'max_age':3,'hungarian':False,'scene_reset_repeat_check':'passed','CPU_association_ms':times,'tracking_metrics':None,'reason':'Complete mini diagnostic, not full official val; no AMOTA claimed','detection_source':str(Path(a.detections).resolve()),'video_fps':2},indent=2))
if __name__=='__main__':main()
