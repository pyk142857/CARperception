"""Measured nuScenes Cylinder3D inference; GT is read only after model prediction."""
import argparse,json,sys,time,subprocess
from pathlib import Path
import numpy as np
from evidence import sha256
from lidarseg_utils import cylinder_features,remap_labels,confusion_matrix,NAMES
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/lidarseg')
    args=p.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    import torch,yaml
    torch.set_num_threads(2);torch.manual_seed(20260924);np.random.seed(20260924)
    repo=ROOT/'third_party/Cylinder3D';sys.path.insert(0,str(repo))
    sys.path[:0]=[str(ROOT/'third_party/Cylinder3D_legacy'),str(ROOT/'third_party/MapTR/mmdetection3d')]
    from builder.model_builder import build
    config=repo/'config/nuScenes.yaml';weight=ROOT/'checkpoints/cylinder3d_nuscenes.pt'
    cfg=yaml.safe_load(config.read_text());model=build(cfg['model_params'])
    state=torch.load(weight,map_location='cpu')
    # Original checkpoint uses polar_* names and SpConv 1 RSCK kernels.
    # Use original SpConv 1 kernels and shared-index behavior; no kernel permutation.
    state={k.replace('polar_','cylinder_',1):v for k,v in state.items()}
    model.load_state_dict(state,strict=True);model.cuda().eval()
    print('All 274 checkpoint tensors strictly loaded',flush=True)
    data=ROOT/'data/nuscenes';meta=data/'v1.0-mini'
    labels={x['token']:x for x in json.loads((meta/'lidarseg.json').read_text())}
    labelmap=repo/'config/label_mapping/nuscenes.yaml';mapping=yaml.safe_load(labelmap.read_text())['learning_map']
    packets=json.loads((ROOT/'manifests/frame_packets.json').read_text())
    if args.limit:packets=packets[:args.limit]
    rows=[];hist=np.zeros((17,17),dtype=np.int64);artifacts={};source_hashes={}
    for i,packet in enumerate(packets):
        path=Path(packet['sensors']['LIDAR_TOP']['path']);points=np.fromfile(path,dtype=np.float32).reshape(-1,5)
        dp=cfg['dataset_params'];grid,features=cylinder_features(points,cfg['model_params']['output_shape'],dp['min_volume_space'],dp['max_volume_space'])
        ft=torch.from_numpy(features).cuda();indices=torch.from_numpy(grid).cuda()
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            logits=model([ft],[indices],1)
            point_logits=logits[0,:,indices[:,0],indices[:,1],indices[:,2]].T
            probability=point_logits.softmax(dim=1);confidence,pred=probability.max(dim=1)
            prediction=pred.cpu().numpy().astype(np.uint8);confidence=confidence.cpu().numpy()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        del logits,point_logits,probability,ft,indices
        # Evaluation/display only: ground truth never enters feature construction or model.
        sd_token=packet['sensors']['LIDAR_TOP']['sample_data_token']
        gtpath=data/labels[sd_token]['filename'];rawgt=np.fromfile(gtpath,dtype=np.uint8)
        if len(rawgt)!=len(points):raise ValueError('Point/GT count mismatch')
        gt=remap_labels(rawgt,mapping);hist+=confusion_matrix(gt,prediction)
        output=args.out/(packet['sample_token']+'.npz')
        np.savez_compressed(output,prediction=prediction,ground_truth=gt,confidence=confidence)
        artifacts[str(output)]=sha256(output);source_hashes[str(path)]=sha256(path);source_hashes[str(gtpath)]=sha256(gtpath)
        rows.append(dict(sample_token=packet['sample_token'],lidar_sample_data_token=sd_token,
                         source='measured',status='passed',point_count=len(points),path=str(output),
                         inference_seconds=elapsed,predicted_class_counts=np.bincount(prediction,minlength=17).tolist()))
        print(f'frame {i+1}/{len(packets)}: {len(points)} points, {elapsed:.3f}s',flush=True)
    union=hist.sum(0)+hist.sum(1)-hist.diagonal()
    iou=np.divide(hist.diagonal(),union,out=np.full(17,np.nan),where=union>0)
    metrics=dict(scope='scene-0061 teaching diagnostic; mini overlaps pretraining; not independent validation',
                 confusion_matrix=hist.tolist(),per_class_iou={NAMES[i]:float(iou[i]) if np.isfinite(iou[i]) else None for i in range(1,17)},
                 mean_iou_present_classes=float(np.nanmean(iou[1:])),point_accuracy=float(hist.diagonal().sum()/hist.sum()),
                 ignored_gt_id=0,valid_points=int(hist.sum()))
    predpath=args.out/'predictions.json';predpath.write_text(json.dumps(rows,indent=2));artifacts[str(predpath)]=sha256(predpath)
    sources=[config,labelmap,weight,Path(__file__),ROOT/'tools/lidarseg_utils.py',ROOT/'tools/prepare_cylinder3d.py',ROOT/'manifests/frame_packets.json']+list((ROOT/'third_party/Cylinder3D_legacy/network').glob('*.py'))+list((ROOT/'third_party/MapTR/mmdetection3d/mmdet3d/ops/spconv').glob('*.py'))
    source_hashes.update({str(f):sha256(f) for f in sources})
    summary=dict(status='passed',source='measured',model='official Cylinder3D nuScenes',frames=len(rows),
                 point_count=sum(r['point_count'] for r in rows),strict_checkpoint=True,checkpoint_conversion='polar_ prefix -> cylinder_; original RSCK kernels',
                 repo_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
                 torch_version=torch.__version__,spconv_version='legacy source bundled with MapTR MMDetection3D',
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),classes=NAMES,metrics=metrics,
                 source_sha256=source_hashes,artifacts=artifacts)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('Saved '+str(predpath),flush=True)

if __name__=='__main__':main()
