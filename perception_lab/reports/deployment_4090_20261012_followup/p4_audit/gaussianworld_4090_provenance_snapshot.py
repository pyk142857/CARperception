#!/usr/bin/env python3
"""Official GaussianWorld streaming inference on mini, with no occupancy labels.

Runs the original FP32 network and voxel grid. Images transfer one frame at a
 time; history_anchor follows the official eval.py loop. Labels never enter
 prediction. Full-scene streaming is distinct from overlapping 30-frame eval.
"""
import argparse, csv, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMS = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
CLASSES = ['noise','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian','traffic_cone','trailer','truck','driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation','empty']

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def transform(rotation,translation):
    import numpy as np
    from pyquaternion import Quaternion
    m=np.eye(4);m[:3,:3]=Quaternion(rotation).rotation_matrix;m[:3,3]=translation
    return m

def prepare(nusc,sample,prev):
    import numpy as np, torch, mmcv
    from dataset.transform_3d import ImageAug3D,NormalizeMultiviewImage,PadMultiViewImage
    sd=nusc.get('sample_data',sample['data']['LIDAR_TOP']);cs=nusc.get('calibrated_sensor',sd['calibrated_sensor_token']);ep=nusc.get('ego_pose',sd['ego_pose_token'])
    l2e=transform(cs['rotation'],cs['translation']);l2g=transform(ep['rotation'],ep['translation'])@l2e
    images=[];projections=[];paths=[]
    for cam in CAMS:
        cd=nusc.get('sample_data',sample['data'][cam]);cc=nusc.get('calibrated_sensor',cd['calibrated_sensor_token']);ce=nusc.get('ego_pose',cd['ego_pose_token'])
        c2g=transform(ce['rotation'],ce['translation'])@transform(cc['rotation'],cc['translation'])
        k=np.eye(4);k[:3,:3]=cc['camera_intrinsic'];projections.append(k@np.linalg.inv(c2g)@l2g)
        path=Path(nusc.dataroot)/cd['filename'];paths.append({'path':str(path),'sha256':digest(path),'timestamp_us':cd['timestamp']});images.append(mmcv.imread(str(path),'unchanged').astype(np.float32))
    data={'img':images}
    for t in [ImageAug3D([864,1600],[1.,1.],False,False),NormalizeMultiviewImage(mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True),PadMultiViewImage(size_divisor=32)]: data=t(data)
    meta={'scene_name':nusc.get('scene',sample['scene_token'])['name'],'lidar2img':[projections],'img_shape':data['img_shape'],'img_aug_matrix':np.stack(data['img_aug_matrix'])[None]}
    if prev is not None:meta['lidar2global']=[prev,l2g]
    imgs=torch.from_numpy(np.stack([x.transpose(2,0,1) for x in data['img']])[None])
    return imgs,[meta],l2g,l2e,paths

def patch_unlabelled(repo):
    """Minimal reproducible patch: only remove label reshape/output dependency."""
    replacements={
        'model/segmentor/gaussian_segmentor_stream.py': [('label = label.repeat(1, anchor.shape[1], 1, 1, 1)', 'label = label.repeat(1, anchor.shape[1], 1, 1, 1) if label is not None else None')],
        'model/head/gaussian_occ_head.py': [('spatial_shape = label.shape[2:]', 'spatial_shape = label.shape[2:] if label is not None else self.gt_xyz.shape[1:4]'), ("'ce_label': label.squeeze(0)", "'ce_label': label.squeeze(0) if label is not None else None")],
    }
    for relative,pairs in replacements.items():
        p=repo/relative;s=p.read_text()
        for original,changed in pairs:
            if changed not in s:
                if original not in s:raise RuntimeError(f'Unsupported source in {p}: {original}')
                s=s.replace(original,changed)
        p.write_text(s)

def main():
    a=argparse.ArgumentParser();a.add_argument('--repo',type=Path,default=ROOT/'third_party/GaussianWorld');a.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/gaussianworld/ckpt_stream.pth');a.add_argument('--output','--out',type=Path,default=ROOT/'runs/deployment_4090_20261008/gaussianworld');a.add_argument('--scene',default='scene-0061');a.add_argument('--packets',type=Path);a.add_argument('--max-frames','--limit',type=int,default=39);a.add_argument('--warmup',type=int,default=1);a.add_argument('--validate-label-independence',action='store_true');args=a.parse_args()
    args.repo=args.repo.resolve();args.checkpoint=args.checkpoint.resolve();args.output=args.output.resolve()
    if args.packets:args.packets=args.packets.resolve()
    args.output.mkdir(parents=True,exist_ok=True);patch_unlabelled(args.repo);sys.path.insert(0,str(args.repo));os.chdir(args.repo)
    import torch,numpy as np
    from mmengine import Config
    from mmengine.registry import MODELS
    from mmengine.runner import set_random_seed
    import model
    from nuscenes.nuscenes import NuScenes
    set_random_seed(1);torch.backends.cudnn.benchmark=True
    cfg=Config.fromfile(str(args.repo/'config/nusc_surroundocc_stream_eval.py'));cfg.model.backbone.init_cfg=None
    start=time.perf_counter();net=MODELS.build(cfg.model)
    ckpt=torch.load(args.checkpoint,map_location='cpu');state=ckpt.get('state_dict',ckpt);state={k.removeprefix('module.'):v for k,v in state.items()};load=net.load_state_dict(state,strict=True);net=net.cuda().eval();torch.cuda.synchronize();load_s=time.perf_counter()-start
    nusc=NuScenes(version='v1.0-mini',dataroot=str(ROOT/'data/nuscenes'),verbose=False)
    scene=next(s for s in nusc.scene if s['name']==args.scene);samples=[];token=scene['first_sample_token']
    while token:s=nusc.get('sample',token);samples.append(s);token=s['next']
    
    if args.packets:
        packets=json.loads(args.packets.read_text());samples=[nusc.get('sample',p['sample_token']) for p in packets]
        assert len({s['scene_token'] for s in samples})==1, 'Use one scene per run; explicit history reset required.'
        assert [s['timestamp'] for s in samples]==sorted(s['timestamp'] for s in samples)
    samples=samples[:args.max_frames];rows=[];records=[];history=None;prev=None
    manifest={'model':'GaussianWorld streaming','source':'measured','checkpoint_sha256':digest(args.checkpoint),'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip(),'config_sha256':digest(args.repo/'config/nusc_surroundocc_stream_eval.py'),'strict_load':str(load),'cold_model_load_s':load_s,'protocol':'scene-stream','amp':False,'image_shape':[1,6,3,864,1600],'gaussians':25600,'grid_shape':[200,200,16],'axis_order':'xyz','coordinate_frame':'LIDAR_TOP','pc_range':[-50,-50,-5,50,50,3],'voxel_size_m':[.5,.5,.5],'classes':CLASSES,'empty_index':17,'ignore_index':0,'evaluation':{'status':'blocked_data','reason':'No SurroundOcc voxel ground truth for mini; no synthetic labels.'},'frames':records}
    try:
        with torch.inference_mode():
            imgs,metas,_,_,_=prepare(nusc,samples[0],None);imgs=imgs.cuda()
            for _ in range(args.warmup):net(imgs=imgs,metas=metas,label=None,history_anchor=None)
            if args.validate_label_independence:
                captured={}
                def capture_head(module, unused, kwargs):
                    captured['anchors']=kwargs['anchors'].clone()
                hook=net.head.register_forward_pre_hook(capture_head,with_kwargs=True)
                first=net(imgs=imgs,metas=metas,label=None,history_anchor=None)['ce_input'];hook.remove()
                repeated=net(imgs=imgs,metas=metas,label=None,history_anchor=None)['ce_input']
                audit={}
                def compare(x,y):
                    return {'max_abs':float((x-y).abs().max()),'mean_abs':float((x-y).abs().mean()),'voxel_class_disagreements':int((x.argmax(1)!=y.argmax(1)).sum()),'num_voxels':640000}
                audit['full_network_repeat']=compare(first,repeated)
                # Same actual predicted anchors isolate the sole label-dependent head branch.
                head_none=net.head(anchors=captured['anchors'],label=None,output_dict={})['ce_input']
                head_labels=net.head(anchors=captured['anchors'],label=torch.zeros((1,1,200,200,16),device='cuda',dtype=torch.long),output_dict={})['ce_input']
                head_repeat=net.head(anchors=captured['anchors'],label=None,output_dict={})['ce_input']
                audit['same_anchors_label_vs_none']=compare(head_none,head_labels)
                audit['same_anchors_repeat']=compare(head_none,head_repeat)
                manifest['label_independence_audit']=audit
                del first,repeated,head_none,head_labels,head_repeat,captured
            del imgs;torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            for i,s in enumerate(samples):
                t0=time.perf_counter();imgs,metas,l2g,l2e,paths=prepare(nusc,s,prev);t1=time.perf_counter();imgs=imgs.cuda();torch.cuda.synchronize();t2=time.perf_counter()
                result=net(imgs=imgs,metas=metas,label=None,history_anchor=history);torch.cuda.synchronize();t3=time.perf_counter();history=result['history_anchor'];pred=result['ce_input'].argmax(1).to(torch.uint8).cpu().numpy()[0];t4=time.perf_counter()
                path=args.output/f'{s["token"]}.npz';np.savez_compressed(path,occupancy=pred,lidar2ego=l2e,lidar2global=l2g)
                row={'frame_id':i,'sample_token':s['token'],'preprocess_ms':(t1-t0)*1000,'h2d_ms':(t2-t1)*1000,'model_ms':(t3-t2)*1000,'postprocess_d2h_ms':(t4-t3)*1000,'e2e_ms':(t4-t0)*1000,'process_gpu_memory_mib':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip(),'allocated_peak_mb':torch.cuda.max_memory_allocated()/1e6,'reserved_peak_mb':torch.cuda.max_memory_reserved()/1e6};rows.append(row)
                records.append({'sample_token':s['token'],'scene_token':s['scene_token'],'timestamp_us':s['timestamp'],'source':'measured','occupancy_path':str(path),'sha256':digest(path),'class_counts':np.bincount(pred.ravel(),minlength=18).tolist(),'inputs':paths,'history_source':None if i==0 else samples[i-1]['token']});prev=l2g
                print(json.dumps(row),flush=True);del result,imgs,pred
                (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        manifest['status']='passed';manifest['frames_completed']=len(records)
    except Exception as e:
        manifest['status']='blocked_resource' if isinstance(e,torch.cuda.OutOfMemoryError) else 'failed';manifest['error']=repr(e);manifest['allocated_peak_mb']=torch.cuda.max_memory_allocated()/1e6;manifest['reserved_peak_mb']=torch.cuda.max_memory_reserved()/1e6
        raise
    finally:
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if rows:
            with open(args.output/'timing.csv','w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
