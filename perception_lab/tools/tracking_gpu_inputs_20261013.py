"""Freeze existing detections and verify real mini LiDAR inputs, without loading GT."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/tracking_gpu_20261013'
SPECS={
 'scene0061':(39,7789,'dd6df55652026f2b04c7b18f9adde44123b664e8cb3706e53d6d30faca344d2f','9680b0d03e49cc78be200a9abe3b99602312123adbf8e12667b9fbbbe7d5c549'),
 'mini_val':(81,16199,'a6fb424ccbe629f3af27172202960c4533cd48ddd887cd201f0d464ee22c56a7','85fba9112c777485268e116fb22d4dcee3a6533de60b70e2421d0abc547b6291')}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def paths(split):
 b=ROOT/'reports/deployment_4090_20261008'
 return b/'bevfusion_reference'/split/'evaluation/official_predictions.json',b/'inputs'/(split+'_packets.json')
def verify():
 rows=[]
 for split,(count,boxes,digest,pdigest) in SPECS.items():
  det,pack=paths(split)
  if sha(det)!=digest or sha(pack)!=pdigest:raise ValueError('Frozen input changed: '+split)
  data=json.loads(det.read_text());packets=json.loads(pack.read_text());tokens=[p['sample_token'] for p in packets]
  if len(tokens)!=count or len(set(tokens))!=count or set(data['results'])!=set(tokens):raise ValueError('Token coverage')
  if sum(map(len,data['results'].values()))!=boxes:raise ValueError('Box count')
  frames=[];last={}
  for p in packets:
   lidar=p['sensors']['LIDAR_TOP'];path=Path(lidar['path']);points=np.fromfile(path,dtype=np.float32).reshape(-1,5)
   if not np.isfinite(points).all():raise ValueError('Nonfinite raw LiDAR')
   transform=np.asarray(lidar['T_ego_to_global'],np.float64)@np.asarray(lidar['T_sensor_to_ego'],np.float64)
   xyz=points[:,:3].astype(np.float64);world=xyz@transform[:3,:3].T+transform[:3,3];inv=np.linalg.inv(transform);restored=world@inv[:3,:3].T+inv[:3,3]
   error=float(np.max(np.abs(restored-xyz)));assert error<1e-8
   timestamp=lidar['timestamp_us']/1e6;scene=p['scene_token'];delta=timestamp-last[scene] if scene in last else None
   if delta is not None and delta<=0:raise ValueError('Scene time not strictly increasing')
   last[scene]=timestamp
   frames.append(dict(sample_token=p['sample_token'],scene_token=scene,timestamp_seconds=timestamp,delta_seconds=delta,raw_lidar_path=str(path),raw_lidar_sha256=sha(path),raw_points=len(points),features='x,y,z,intensity,ring; remission/intensity column3',sensor_to_global=transform.tolist(),transform_sha256=hashlib.sha256(transform.tobytes()).hexdigest(),roundtrip_max_abs_m=error,raw_detection_count=len(data['results'][p['sample_token']])))
  rows.append(dict(split=split,frames=count,detection_count=boxes,detection_path=str(det.relative_to(ROOT)),detection_sha256=digest,packets_path=str(pack.relative_to(ROOT)),packets_sha256=pdigest,point_count=sum(x['raw_points'] for x in frames),samples=frames))
 result=dict(status='passed',source='frozen BEVFusion native global results; no detector rerun',gt_loaded=False,splits=rows)
 REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'input_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
 state=json.loads((REPORT/'status.json').read_text());state['stages']['inputs_frozen']='passed';(REPORT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps({x['split']:{'frames':x['frames'],'boxes':x['detection_count'],'points':x['point_count']} for x in rows}))
 return result
if __name__=='__main__':verify()
