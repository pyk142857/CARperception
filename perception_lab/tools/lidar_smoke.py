"""Native MMDetection3D mini inference with upstream infos and strict weights."""
import argparse,json,os,sys,copy,hashlib
from pathlib import Path
import numpy as np
import torch
from evidence import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['pointpillars','centerpoint'],required=True);p.add_argument('--packets',required=True);p.add_argument('--sample-index',type=int,default=0);p.add_argument('--out',required=True);args=p.parse_args()
 torch.set_num_threads(2)
 torch.manual_seed(20260922);np.random.seed(20260922)
 root=Path(__file__).resolve().parents[1];out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 os.chdir(root)
 repo=root/'third_party/mmdetection3d';sys.path.insert(0,str(repo))
 import mmengine
 from mmengine.dataset import Compose,pseudo_collate
 from mmengine.runner import load_checkpoint
 from mmdet3d.apis import init_model
 from mmdet3d.utils import register_all_modules
 from mmdet3d.structures import get_box_type
 from tools.dataset_converters.nuscenes_converter import create_nuscenes_infos
 from tools.dataset_converters.update_infos_to_v2 import update_nuscenes_infos
 from pyquaternion import Quaternion
 register_all_modules()
 asset=json.loads((root/'configs/locked/lidar_assets.json').read_text())[args.model]
 cfg=repo/asset['config'];weight=root/'checkpoints'/asset['file']
 repo_commit=json.loads((root/'configs/locked/repositories.json').read_text())['mmdetection3d']['commit']
 cache_key=hashlib.sha256(json.dumps({'version':'v1.0-mini','split':'mini_train+mini_val','commit':repo_commit,'max_sweeps':10,'converter_sha':sha256(repo/'tools/dataset_converters/nuscenes_converter.py'),'updater_sha':sha256(repo/'tools/dataset_converters/update_infos_to_v2.py')},sort_keys=True).encode()).hexdigest()[:16]
 cache=root/'data/infos'/('mmdet3d_'+cache_key);cache.mkdir(parents=True,exist_ok=True)
 old=cache/'original';new=cache/'v2';old.mkdir(exist_ok=True);new.mkdir(exist_ok=True)
 files=[new/('mini_infos_'+s+'.pkl') for s in ['train','val']]
 if not all(x.is_file() for x in files):
  create_nuscenes_infos(str(root/'data/nuscenes'),str(old/'mini'),version='v1.0-mini',max_sweeps=10)
  for split in ['train','val']:update_nuscenes_infos(str(old/('mini_infos_'+split+'.pkl')),str(new))
 packet=json.loads(Path(args.packets).read_text())[args.sample_index]
 records=[r for f in files for r in mmengine.load(f)['data_list']]
 info=next(x for x in records if x['token']==packet['sample_token'])
 model=init_model(str(cfg),str(weight),device='cuda:0')
 load_checkpoint(model,str(weight),map_location='cpu',strict=True)
 model.cfg.dump(str(out/'resolved_config.py'))
 pipeline=Compose(copy.deepcopy(model.cfg.test_dataloader.dataset.pipeline))
 box_type,box_mode=get_box_type(model.cfg.test_dataloader.dataset.box_type_3d)
 lidar=copy.deepcopy(info['lidar_points']);lidar['lidar_path']=str(root/'data/nuscenes/samples/LIDAR_TOP'/Path(lidar['lidar_path']).name)
 # Explicit allowlist: no annotations or evaluator-only fields reach the network.
 inputs=dict(lidar_points=lidar,lidar_sweeps=copy.deepcopy(info.get('lidar_sweeps',[])),timestamp=info['timestamp'],box_type_3d=box_type,box_mode_3d=box_mode,token=info['token'])
 for s in inputs['lidar_sweeps']:
  path=Path(s['lidar_points']['lidar_path'])
  if not path.is_absolute():s['lidar_points']['lidar_path']=str(root/path)
 data=pipeline(inputs)
 with torch.inference_mode(): result=model.test_step(pseudo_collate([data]))[0]
 pred=result.pred_instances_3d.to('cpu');boxes=pred.bboxes_3d
 raw=boxes.tensor.numpy();scores=pred.scores_3d.numpy();labels=pred.labels_3d.numpy()
 if not np.isfinite(raw).all() or not np.isfinite(scores).all() or not (boxes.dims.numpy()>0).all():raise ValueError('Invalid native boxes')
 np.savez_compressed(out/'native_predictions.npz',boxes=raw,scores=scores,labels=labels)
 T=np.array(packet['sensors']['LIDAR_TOP']['T_sensor_to_ego']);rotation=Quaternion(matrix=T[:3,:3])
 centers=boxes.gravity_center.numpy();dims=boxes.dims.numpy();yaw=boxes.yaw.numpy();standard=[];roundtrip=[]
 names=model.dataset_meta['classes']
 for i in range(len(scores)):
  center=T[:3,:3]@centers[i]+T[:3,3]
  orientation=rotation*Quaternion(axis=[0,0,1],radians=float(yaw[i]))
  velocity=T[:3,:3]@np.array([raw[i,7],raw[i,8],0]) if raw.shape[1]>=9 else None
  standard.append(dict(center_xyz=center.tolist(),size_wlh=dims[i,[1,0,2]].tolist(),rotation_wxyz=orientation.elements.tolist(),velocity_xy=velocity[:2].tolist() if velocity is not None else None,class_name=names[int(labels[i])],score=float(scores[i])))
  reconstructed=np.linalg.inv(T)@np.r_[center,1];roundtrip.append(float(np.max(np.abs(reconstructed[:3]-centers[i]))))
 assert len(standard)==len(scores)
 assert max(roundtrip,default=0)<1e-5
 record=dict(run_id=out.parts[-3],module_id='M04' if args.model=='pointpillars' else 'M05',model_id=args.model,scope='mini_single_frame_smoke',source='measured',status='passed',sample_token=packet['sample_token'],scene_token=packet['scene_token'],timestamp_us=packet['timestamp_us'],native_coordinate_system='LiDAR bottom-centre xyz length-width-height yaw',label_namespace='nuScenes_detection_10',source_config_hash=sha256(out/'resolved_config.py'),checkpoint_sha256=sha256(weight),boxes3d=standard)
 (out/'predictions.json').write_text(json.dumps([record],indent=2))
 (out/'summary.json').write_text(json.dumps({'source':'measured','scope':'mini_single_frame_smoke','sample_count':1,'box_count':len(standard),'adapter_center_roundtrip_error':max(roundtrip,default=0),'input_sweeps_available':len(inputs['lidar_sweeps']),'infos_cache':str(cache),'infos_key':cache_key,'strict_checkpoint_load':True,'device':'cuda:0','evaluation':None,'reason':'Mini smoke only; official trainval unavailable'},indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Polygon
 points=data['inputs']['points'].numpy()
 fig,ax=plt.subplots(figsize=(9,9));ax.scatter(points[::5,0],points[::5,1],s=.1,c='gray')
 corners=boxes.corners.numpy()
 for i in range(len(scores)):
  if scores[i]<.25:continue
  poly=corners[i,[0,3,7,4],:2];ax.add_patch(Polygon(poly,fill=False,edgecolor='red',linewidth=.7))
 ax.set(xlim=(-55,55),ylim=(-55,55),xlabel='LiDAR x (m)',ylabel='LiDAR y (m)',title=args.model+' measured predictions (display score >= 0.25)');ax.set_aspect('equal')
 fig.savefig(out/'bev_predictions.png',dpi=130);plt.close(fig)
if __name__=='__main__':main()
