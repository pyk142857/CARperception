"""Rebuild deployment inputs from official mini tables, independent of old paths."""
import argparse,json,hashlib,subprocess,platform,shutil,sys
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import transform_matrix
ROOT=Path(__file__).resolve().parents[1]
CAMERAS=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 data=ROOT/'data/nuscenes';n=NuScenes(version='v1.0-mini',dataroot=str(data),verbose=False);files=set()
 def sensor(token):
  sd=n.get('sample_data',token);cs=n.get('calibrated_sensor',sd['calibrated_sensor_token']);ep=n.get('ego_pose',sd['ego_pose_token']);p=data/sd['filename']
  assert p.is_file(),p;files.add(p)
  return dict(path=str(p),relative_path=sd['filename'],sample_data_token=token,timestamp_us=sd['timestamp'],K=cs['camera_intrinsic'],width=sd['width'],height=sd['height'],T_sensor_to_ego=transform_matrix(cs['translation'],Quaternion(cs['rotation'])).tolist(),T_ego_to_global=transform_matrix(ep['translation'],Quaternion(ep['rotation'])).tolist())
 def packets(names):
  result=[]
  for scene in sorted([x for x in n.scene if x['name'] in names],key=lambda x:x['name']):
   token=scene['first_sample_token'];count=0
   while token:
    sample=n.get('sample',token);sensors={c:sensor(sample['data'][c]) for c in CAMERAS+['LIDAR_TOP']};ref=sensors['LIDAR_TOP'];sweeps=[]
    prev=n.get('sample_data',sample['data']['LIDAR_TOP'])['prev']
    for _ in range(10):
     if not prev:break
     item=sensor(prev);item['dt_to_reference_seconds']=(ref['timestamp_us']-item['timestamp_us'])/1e6
     item['T_sensor_to_ego_ref']=(np.linalg.inv(ref['T_ego_to_global'])@np.asarray(item['T_ego_to_global'])@np.asarray(item['T_sensor_to_ego'])).tolist();sweeps.append(item);prev=n.get('sample_data',prev)['prev']
    result.append(dict(sample_token=token,scene_token=scene['token'],scene_name=scene['name'],timestamp_us=sample['timestamp'],reference_timestamp_us=ref['timestamp_us'],camera_order=CAMERAS,sensors=sensors,lidar_sweeps=sweeps,lidar_raw_fields=['x','y','z','intensity','ring']))
    token=sample['next'];count+=1
   assert count==scene['nbr_samples']
  assert len({r['sample_token'] for r in result})==len(result)
  for first,second in zip(result,result[1:]):
   if first['scene_token']==second['scene_token']:assert first['timestamp_us']<second['timestamp_us']
  return result
 split_names=create_splits_scenes()['mini_val'];scene=packets(['scene-0061']);val=packets(split_names);assert len(scene)==39
 for name,rows in [('scene0061',scene),('mini_val',val)]:
  (a.out/(name+'_packets.json')).write_text(json.dumps(rows,indent=2)+'\n')
  (a.out/(name+'_samples.json')).write_text(json.dumps([dict(sample_token=r['sample_token'],scene_token=r['scene_token'],scene_name=r['scene_name'],timestamp_us=r['timestamp_us']) for r in rows],indent=2)+'\n')
 for p in (data/'v1.0-mini').glob('*.json'):files.add(p)
 hashes={str(p.relative_to(data)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)}
 (a.out/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
 hardware={}
 for key,cmd in [('nvidia_smi',['nvidia-smi']),('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv']),('compiler',['gcc','--version']),('cuda_118',['/usr/local/cuda-11.8/bin/nvcc','--version']),('cuda_122',['/usr/local/cuda-12.2/bin/nvcc','--version'])]:
  p=subprocess.run(cmd,capture_output=True,text=True);hardware[key]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 hardware.update(platform=platform.platform(),python=sys.version,disk_free_bytes=shutil.disk_usage(ROOT).free,git_start_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
 (a.out/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
 summary=dict(status='passed',scene0061_frames=len(scene),mini_val_frames=len(val),mini_val_scenes=split_names,camera_order=CAMERAS,file_count=len(files),files_bytes=sum(x['bytes'] for x in hashes.values()),inputs_sha256={p.name:digest(p) for p in a.out.glob('*packets.json')},label_scope='official mini metadata; inference packets contain no annotations; pretraining overlap possible')
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
