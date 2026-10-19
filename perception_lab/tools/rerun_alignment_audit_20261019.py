"""Audit frozen-record display center against saved native tensors; no inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tracking_gpu_visualize_20261013 import load_inputs,frozen_display_boxes,ego_boxes,ROOT,VISUAL_REPORT
from deployment_visualize import camera_tile,CAMERAS,transform_points

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 VISUAL_REPORT.mkdir(parents=True,exist_ok=True)
 result={'scope':'Display compatibility only; all historical detections/tracks/evaluation unchanged','splits':[]}
 for split in ['scene0061','mini_val']:
  packets,artifacts,hashes=load_inputs(split)
  native_path=ROOT/f'reports/deployment_4090_20261008/bevfusion_reference/{split}/native_predictions.json'
  native={x['sample_token']:x for x in json.loads(native_path.read_text())}
  error=0.;checked=0;camera_deltas=[]
  for packet in packets:
   token=packet['sample_token'];row=native[token];raw=np.asarray(row['tensor']);keep=np.asarray(row['scores'])>=.25
   expected=transform_points(raw[keep,:3],np.asarray(packet['sensors']['LIDAR_TOP']['T_sensor_to_ego']))
   actual=np.asarray([x['center_xyz'] for x in frozen_display_boxes(artifacts['bev_detection'][token],packet,'bev_detection')]).reshape(-1,3)
   assert actual.shape==expected.shape
   if len(actual):error=max(error,float(np.max(np.abs(expected-actual))))
   checked+=len(actual)
   for cam in CAMERAS:camera_deltas.append(packet['sensors'][cam]['timestamp_us']-packet['timestamp_us'])
  # Historical gravity_center addition used float32; undoing it retains rounding.
  print('Native center error',split,error,flush=True);assert error<1e-6
  for frame in [0,len(packets)//2,len(packets)-1]:
   packet=packets[frame];rows=artifacts['grae'][packet['sample_token']];pose=np.asarray(packet['sensors']['LIDAR_TOP']['T_ego_to_global'])
   for mode,boxes in [('before',ego_boxes(rows,pose,'grae')),('after',frozen_display_boxes(rows,packet,'grae'))]:
    image=Image.new('RGB',(1278,480))
    for i,cam in enumerate(CAMERAS):image.paste(camera_tile(packet,cam,boxes,'grae'),((i%3)*426,(i//3)*240))
    image.save(VISUAL_REPORT/f'{split}_{frame:02d}_{mode}.jpg',quality=92)
  result['splits'].append(dict(split=split,frames=len(packets),native_centers_checked=checked,max_abs_ego_center_error_m=error,source_hashes=hashes,native_sha256=sha(native_path),camera_time_offset_us_min=min(camera_deltas),camera_time_offset_us_max=max(camera_deltas)))
 # Independent GT diagnostic, never used in the correction: nearest XY matches.
 packet=load_inputs('scene0061')[0][19];pose=np.asarray(packet['sensors']['LIDAR_TOP']['T_ego_to_global'])
 anns=json.loads((ROOT/'data/nuscenes/v1.0-mini/sample_annotation.json').read_text())
 gt=[dict(x,detection_score=1,detection_name='gt') for x in anns if x['sample_token']==packet['sample_token']]
 gt=np.asarray([x['center_xyz'] for x in ego_boxes(gt,pose,'gt')])
 rows=load_inputs('scene0061')[1]['bev_detection'][packet['sample_token']]
 old=ego_boxes(rows,pose,'bev_detection');new=frozen_display_boxes(rows,packet,'bev_detection')
 xyz=np.asarray([b['center_xyz'] for b in old]);dist=np.linalg.norm(xyz[:,None,:2]-gt[None,:,:2],axis=2);a,b=linear_sum_assignment(dist)
 selected=[(i,j) for i,j in zip(a,b) if dist[i,j]<.6 and old[i]['score']>.4]
 result['gt_diagnostic_frame19']=dict(matching='XY nearest assignment <0.6m, score>0.4; diagnostic not official metric or calibration fitting',pairs=len(selected),median_z_error_before_m=float(np.median([old[i]['center_xyz'][2]-gt[j,2] for i,j in selected])),median_z_error_after_m=float(np.median([new[i]['center_xyz'][2]-gt[j,2] for i,j in selected])))
 result['status']='passed';(VISUAL_REPORT/'alignment_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['gt_diagnostic_frame19']))
if __name__=='__main__':main()
