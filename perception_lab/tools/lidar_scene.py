"""Complete mini scene diagnostic, reusing the verified native per-frame entry."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import cv2

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['centerpoint','pointpillars'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 packets=json.loads((root/'manifests/frame_packets.json').read_text());records=[];index=[];start=time.perf_counter()
 for i,packet in enumerate(packets):
  frame=out/'frames'/f'{i:06d}';frame.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,str(root/'tools/lidar_smoke.py'),'--model',a.model,'--packets',str(root/'manifests/frame_packets.json'),'--sample-index',str(i),'--out',str(frame)]
  with (frame/'command.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  pred=json.loads((frame/'predictions.json').read_text())[0]
  assert pred['sample_token']==packet['sample_token']
  pred['run_id']=out.relative_to(root/'runs').parts[0];pred['scope']='mini_scene_engineering_only';records.append(pred)
  index.append({'sample_token':packet['sample_token'],'scene_token':packet['scene_token'],'timestamp_us':packet['timestamp_us'],'source_interval_seconds':0 if i==0 else (packet['timestamp_us']-packets[i-1]['timestamp_us'])/1e6,'prediction_path':str(frame/'predictions.json'),'image_path':str(frame/'bev_predictions.png')})
  print(i+1,'/',len(packets),packet['sample_token'],'passed',flush=True)
 if [x['sample_token'] for x in records]!=[x['sample_token'] for x in packets]:raise ValueError('Scene coverage mismatch')
 first=cv2.imread(index[0]['image_path']);h,w=first.shape[:2];video=cv2.VideoWriter(str(out/'mini_centerpoint_bev.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(w,h))
 if not video.isOpened():raise RuntimeError('MP4 writer unavailable')
 for row in index:
  image=cv2.imread(row['image_path']);cv2.putText(image,'MINI diagnostic | '+row['sample_token'][:12],(20,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2);video.write(image)
 video.release()
 (out/'predictions.json').write_text(json.dumps(records,indent=2));(out/'frame_index.json').write_text(json.dumps(index,indent=2))
 (out/'summary.json').write_text(json.dumps({'source':'measured','scope':'mini_scene_engineering_only','sample_count':len(records),'model':a.model,'wall_seconds':time.perf_counter()-start,'timing_scope':'includes process and model reload per frame, inference and rendering; not warm benchmark','video_fps':2,'full_official_val':False},indent=2))
if __name__=='__main__':main()
