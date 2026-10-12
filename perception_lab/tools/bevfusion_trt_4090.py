"""Run real CUDA-BEVFusion with the pinned reference CPU preprocessing.

Launch after sourcing tools/bevfusion_trt_env.sh. No annotation is read.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from bevfusion_4090 import digest, write_json, packet_input, make_pipeline, standardize

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/deployment_4090.yaml')
    p.add_argument('--packets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--capture-pooling', type=Path)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--paired-pooling-rounds',type=int,default=0)
    a = p.parse_args()
    cfg_deploy = json.loads(a.config.read_text())
    bcfg, tcfg = cfg_deploy['bevfusion'], cfg_deploy['tensorrt']
    source = ROOT / bcfg['source']
    model_dir = ROOT / tcfg['model_dir']
    build_dir = ROOT / tcfg['source'] / 'CUDA-BEVFusion/build'
    sys.path[:0] = [str(source), str(build_dir)]
    a.out.mkdir(parents=True, exist_ok=True)
    packets = json.loads(a.packets.read_text())
    if a.limit is not None:
        packets = packets[:a.limit]
    if not packets:
        raise ValueError('Empty input')
    identity = dict(config=digest(a.config), packets=digest(a.packets), adapter=digest(__file__),
                    checkpoint=digest(ROOT/bcfg['checkpoint']), limit=a.limit,
                    native_library=digest(build_dir/'libpybev.so'),
                    core_library=digest(build_dir/'libbevfusion_core.so'),
                    pooling=os.environ.get('CAR_BEV_POOLING', 'upstream'))
    identity['engines'] = {name: digest(model_dir/'build'/(name+'.plan')) for name in ['camera.backbone', 'camera.vtransform', 'fuser', tcfg.get('head_engine','head.bbox')]}
    identity['onnx'] = {str(f.name): digest(f) for f in sorted(model_dir.glob('*.onnx'))}
    start = time.perf_counter()
    import torch
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
    import libpybev
    configs.load(str(ROOT/bcfg['config']), recursive=True)
    cfg = Config(recursive_eval(configs))
    pipeline, steps = make_pipeline(cfg)
    def prepare(packet):
        item = packet_input(packet)
        item.update(box_type_3d=LiDARInstance3DBoxes, box_mode_3d=Box3DMode.LIDAR)
        item = pipeline(item)
        def array(key):
            value = item[key]
            if hasattr(value, 'data'):
                value = value.data
            if hasattr(value, 'numpy'):
                value = value.numpy()
            return np.ascontiguousarray(value, dtype=np.float32)
        images = array('img')[None]
        points = array('points')
        if not 0 < len(points) <= tcfg['max_input_points']:
            raise ValueError('Point capacity exceeded; do not silently truncate')
        matrices = [array(k)[None] for k in ['camera2lidar','camera_intrinsics','lidar2image','img_aug_matrix']]
        return images, points, matrices
    core = libpybev.load_bevfusion(str(model_dir/'build/camera.backbone.plan'),
        str(model_dir/'build/camera.vtransform.plan'), str(model_dir/'lidar.backbone.xyz.onnx'),
        str(model_dir/'build/fuser.plan'), str(model_dir/'build'/(tcfg.get('head_engine','head.bbox')+'.plan')), 'fp16')
    if core is None:
        raise RuntimeError('CUDA-BEVFusion load failed')
    core.print()
    if a.profile: core.set_timer(True)
    cold_s = time.perf_counter()-start
    first = prepare(packets[0])
    for _ in range(a.warmup):
        core.update(*first[2])
        core.forward(first[0].astype(np.float16), first[1].astype(np.float16), False, False)
    torch.cuda.synchronize()
    predictions, native, timings, audits, inputs = [], [], [], [], []
    for index, packet in enumerate(packets):
        t0 = time.perf_counter()
        images, points, matrices = prepare(packet)
        images_half, points_half = images.astype(np.float16), points.astype(np.float16)
        t1 = time.perf_counter()
        core.update(*matrices)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        raw = np.asarray(core.forward(images_half, points_half, False, False)).copy()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if raw.ndim != 2 or raw.shape[1] != 11 or not np.isfinite(raw).all():
            raise ValueError('Invalid backend output')
        detection = dict(boxes_3d=LiDARInstance3DBoxes(torch.tensor(raw[:,:9]), box_dim=9),
                         scores_3d=torch.tensor(raw[:,10]), labels_3d=torch.tensor(raw[:,9].astype(np.int64)))
        boxes, audit = standardize(detection, packet, cfg.object_classes)
        rec = {k:packet[k] for k in ['sample_token','scene_token','timestamp_us']}
        rec.update(source='measured', model='bevfusion_trt', boxes3d=boxes)
        predictions.append(rec)
        native.append(dict(sample_token=packet['sample_token'], tensor=raw.tolist()))
        audits.append(audit)
        t4 = time.perf_counter()
        memory = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        own = [int(line.split(',')[1]) for line in memory.splitlines() if line.split(',')[0].strip() == str(os.getpid())]
        timings.append(dict(frame=index, sample_token=packet['sample_token'], points=len(points),
            preprocess_ms=(t1-t0)*1000, geometry_update_ms=(t2-t1)*1000,
            forward_h2d_model_d2h_ms=(t3-t2)*1000, standardize_ms=(t4-t3)*1000,
            end_to_end_ms=(t4-t0)*1000, process_gpu_mib=max(own,default=0)))
        import hashlib
        inputs.append(dict(sample_token=packet['sample_token'], points=len(points), image_shape=list(images.shape),
            fp32_image_sha256=hashlib.sha256(images.tobytes()).hexdigest(),
            fp32_points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
            matrices_sha256=[hashlib.sha256(m.tobytes()).hexdigest() for m in matrices]))
        if a.capture_pooling and index in {0,len(packets)//2,len(packets)-1}:
            a.capture_pooling.mkdir(parents=True,exist_ok=True)
            os.environ['CAR_BEV_POOL_CAPTURE']=str((a.capture_pooling/('frame_%d'%index)).resolve())
            core.forward(images_half,points_half,False,False)
            torch.cuda.synchronize()
            os.environ.pop('CAR_BEV_POOL_CAPTURE')
        write_json(a.out/'predictions.partial.json', predictions)
        print(json.dumps(timings[-1]),flush=True)
    write_json(a.out/'predictions.json', predictions)
    write_json(a.out/'native_predictions.json', native)
    write_json(a.out/'coordinate_audit.json', audits)
    write_json(a.out/'actual_inputs.json', inputs)
    with (a.out/'timings.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(timings[0]));writer.writeheader();writer.writerows(timings)
    summary={k:dict(mean=float(np.mean([r[k] for r in timings])),p50=float(np.percentile([r[k] for r in timings],50)),p95=float(np.percentile([r[k] for r in timings],95)),first=timings[0][k]) for k in timings[0] if k.endswith('_ms')}
    write_json(a.out/'performance.json',dict(frames=len(packets),cold_start_seconds=cold_s,warmup=a.warmup,timings=summary,
        process_peak_mib=max(r['process_gpu_mib'] for r in timings),
        precision='fp16', boundaries='forward includes image/points H2D, CUDA/TRT, box D2H; geometry update separate; process memory sampled after each frame; rendering excluded'))
    if a.paired_pooling_rounds:
        from bev_pooling_full_model import paired_forwards
        paired_forwards(core, *first, a.paired_pooling_rounds, cfg_deploy['pooling']['warmup'], a.out)
    files=['predictions.json','native_predictions.json','coordinate_audit.json','actual_inputs.json','timings.csv','performance.json']
    write_json(a.out/'manifest.json',dict(status='passed',source='measured',identity=identity,frames=len(packets),
        outputs_sha256={name:digest(a.out/name) for name in files},accuracy_status='not_evaluated'))

if __name__ == '__main__':
    main()
