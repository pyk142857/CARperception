"""Pinned HARP-NeXt mini adapter and original-points-only evaluation.

CPU preprocessing can be audited without checkpoint. Inference is fail-closed:
requires complete author checkpoint['net'] and strict loading, never dummy weights.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/HARP-NeXt_2025'
OUT = ROOT / 'reports/perception_2025_20261013/harpnext'
COMMIT = '532df2659700ae870c0e0d4137a4d35bcc9903b9'
NAMES = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
         'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
         'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
MAPPING = np.array([0,0,7,7,7,0,7,0,0,1,0,0,8,0,2,3,3,4,5,0,0,6,9,10,11,12,13,14,15,0,16,0])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def map_labels(raw):
    raw = np.asarray(raw)
    if raw.ndim != 1 or raw.dtype.kind not in 'ui' or np.any((raw < 0) | (raw >= 32)):
        raise ValueError('Raw nuScenes labels must be integers in 0..31')
    mapped = MAPPING[raw] - 1
    mapped[mapped < 0] = 255
    return mapped


def original_predictions(prediction, original_index, original_count):
    prediction, original_index = np.asarray(prediction), np.asarray(original_index)
    if prediction.ndim != 1 or prediction.shape != original_index.shape:
        raise ValueError('Prediction/index shape mismatch')
    if original_index.dtype.kind not in 'iu' or np.any((original_index < -1) | (original_index >= original_count)):
        raise ValueError('Invalid original-point mapping')
    keep = original_index >= 0
    if not np.array_equal(np.sort(original_index[keep]), np.arange(original_count)):
        raise ValueError('Every original point must occur exactly once')
    restored = np.empty(original_count, dtype=prediction.dtype)
    restored[original_index[keep]] = prediction[keep]
    return restored


def confusion(gt, prediction):
    gt, prediction = np.asarray(gt), np.asarray(prediction)
    if gt.ndim != 1 or gt.shape != prediction.shape or gt.dtype.kind not in 'iu' or prediction.dtype.kind not in 'iu':
        raise ValueError('Invalid point labels/alignment')
    if np.any(((gt < 0) | (gt > 15)) & (gt != 255)) or np.any((prediction < 0) | (prediction > 15)):
        raise ValueError('Expected 0..15 predictions, 0..15 or ignore255 GT')
    keep = gt != 255
    return np.bincount(16 * gt[keep] + prediction[keep], minlength=256).reshape(16, 16)


def summarize(hist):
    hist = np.asarray(hist)
    union = hist.sum(0) + hist.sum(1) - hist.diagonal()
    ious = [float(hist[i,i] / union[i]) if union[i] else None for i in range(16)]
    return {'protocol': 'original-points-only; ignore=255; union>0 macro IoU',
            'valid_points': int(hist.sum()), 'accuracy': float(np.trace(hist)/hist.sum()) if hist.sum() else None,
            'miou_union_gt_zero': float(np.mean([x for x in ious if x is not None])) if any(union) else None,
            'confusion': hist.tolist(), 'classes': [dict(name=name, network_id=i, display_id=i+1,
                intersection=int(hist[i,i]), union=int(union[i]), iou=ious[i]) for i,name in enumerate(NAMES)]}


def mini_rows(data):
    meta = data / 'v1.0-mini'
    read = lambda name: json.loads((meta / (name + '.json')).read_text())
    scenes = {r['token']: r['name'] for r in read('scene')}
    samples = {r['token']: r for r in read('sample')}
    labels = {r['token']: r for r in read('lidarseg')}
    rows = []
    for sd in read('sample_data'):
        if sd['token'] not in labels:
            continue
        sample = samples[sd['sample_token']]
        point_path, label_path = data/sd['filename'], data/labels[sd['token']]['filename']
        if point_path.stat().st_size % 20:
            raise ValueError('Malformed XYZI/ring point file')
        n = point_path.stat().st_size // 20
        gt = np.fromfile(label_path, dtype=np.uint8)
        if n != len(gt):
            raise ValueError('Original point / label count mismatch')
        mapped = map_labels(gt)
        rows.append(dict(sample_token=sample['token'], sample_data_token=sd['token'],
            scene=scenes[sample['scene_token']], timestamp=sample['timestamp'],
            point_path=sd['filename'], label_path=labels[sd['token']]['filename'],
            original_points=n, valid_points=int((mapped != 255).sum()),
            point_sha256=sha256(point_path), label_sha256=sha256(label_path)))
    return sorted(rows, key=lambda r: (r['scene'], r['timestamp']))


def source_check():
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise ValueError('HARP source commit mismatch')
    if not np.array_equal(np.load(SOURCE/'datasets/nuscenes/nuscenes_class_mapping.npy'), MAPPING):
        raise ValueError('Author class mapping mismatch')
    sys.path.insert(0, str(SOURCE))


def processor(rows, data, gpu=False, labels_mode='zero'):
    from datasets.pc_processors.harpnext_pcprocessor import HARPNeXtPCProcessor
    class MiniProcessor(HARPNeXtPCProcessor):
        def __len__(self):
            return len(rows)
        def load_pc(self, index):
            row = rows[index]
            points = np.fromfile(data/row['point_path'], dtype=np.float32).reshape(-1, 5)[:, :4]
            # Inference uses no GT. Real labels are read only in diagnostic/evaluation paths.
            if labels_mode == 'actual':
                raw = np.fromfile(data/row['label_path'], dtype=np.uint8)
            elif labels_mode == 'permuted':
                raw = np.arange(len(points), dtype=np.int64) % 32
            else:
                raw = np.zeros(len(points), dtype=np.uint8)
            return points, map_labels(raw), raw, row['point_path'], row['label_path'], row['sample_data_token']
    previous = Path.cwd()
    try:
        os.chdir(SOURCE)  # Author LaserScan opens its dataset YAML relative to cwd.
        return MiniProcessor(dataset='nuscenes', rootdir=str(data), input_feat=['intensity'], phase='val',
            tta=False, train_augmentations=None, instance_cutmix=False, range_H=32, range_W=480,
            fov_up=10., fov_down=-30., batch_size=1, preproc_gpu=gpu, rank=0)
    finally:
        os.chdir(previous)


def preflight(data, out):
    source_check()
    rows = mini_rows(data)
    out.mkdir(parents=True, exist_ok=True)
    save(out/'mini_frames.json', rows)
    split_rows = {'scene0061': [r for r in rows if r['scene'] == 'scene-0061'],
                  'mini_val': [r for r in rows if r['scene'] in ['scene-0103', 'scene-0916']], 'mini_all': rows}
    for name, selected in split_rows.items():
        # Same schema [relative points, relative labels, sample_data token], actual mini length.
        np.savez(out/(name+'_nuscenes_files.npz'), val=np.array([
            [r['point_path'], r['label_path'], r['sample_data_token']] for r in selected]))
    sample_rows = [split_rows['scene0061'][i] for i in [0,19,38]]
    comparisons = []
    import torch
    for row in sample_rows:
        batches = [processor([row], data, labels_mode=mode).process_batch_cpu(0)
                   for mode in ['zero', 'actual', 'permuted']]
        first, points = batches[0]
        original = np.fromfile(data/row['point_path'], dtype=np.float32).reshape(-1,5)[:,:4]
        assert np.array_equal(points[:len(original)], original)
        for batch, candidate in batches[1:]:
            assert np.array_equal(candidate, points)
            assert torch.equal(first['voxels'], batch['voxels'])
            assert torch.equal(first['coors'], batch['coors'])
        comparisons.append(dict(sample_token=row['sample_token'], original_points=len(original),
            appended_points=len(points)-len(original), gt_variants=3, features_equal=True,
            original_prefix_exact=True, model_predictions_checked=False,
            native_manager_would_consume_processed_labels=len(first['pt_labels'][0])))
    from utils.transformations.transforms import RangeInterpolation, RangeInterpolationCUDA
    # Execute the vectorized CUDA implementation on CPU solely to isolate boundary semantics.
    pixels = np.array([4, 7, 1])
    yaw = ((pixels + .5) / 8 * 2 - 1) * np.pi
    fixture = np.stack([10*np.cos(yaw), -10*np.sin(yaw), np.zeros(3), np.ones(3)], 1).astype(np.float32)
    reference, _ = RangeInterpolation(H=1, W=8, fov_up=10, fov_down=-30)(fixture.copy(), np.zeros(3, dtype=np.int64))
    candidate, _ = RangeInterpolationCUDA(H=1, W=8, fov_up=10, fov_down=-30, device='cpu')(torch.from_numpy(fixture.copy()), torch.zeros(3, dtype=torch.int32))
    boundary = dict(device='cpu', purpose='algorithm boundary probe, not GPU performance',
        original_points=3, cpu_output_points=len(reference), vectorized_output_points=len(candidate),
        equal=bool(np.array_equal(reference, candidate.numpy())),
        note='torch.roll wraps columns; CPU refuses interpolation at first and last columns; both use proj_idx>0')
    zero_pixels = np.array([3, 5, 0])
    zero_yaw = ((zero_pixels + .5) / 8 * 2 - 1) * np.pi
    zero_fixture = np.stack([10*np.cos(zero_yaw), -10*np.sin(zero_yaw), np.zeros(3), np.ones(3)], 1).astype(np.float32)
    zero_op = RangeInterpolation(H=1, W=8, fov_up=10, fov_down=-30)
    zero_a, _ = zero_op(zero_fixture.copy(), np.zeros(3, dtype=np.int64))
    zero_b, _ = zero_op(zero_fixture[[2,0,1]].copy(), np.zeros(3, dtype=np.int64))
    point_zero = dict(original_points=3, output_points_before_reorder=len(zero_a),
        output_points_after_reorder=len(zero_b), same_geometric_input=True,
        note='Which point has index0 changes interpolation; index0 occupancy is false')
    result = dict(status='passed', source='new_measured', inference_frames=0,
        synthetic_boundary_probe=boundary, point_index_zero_probe=point_zero,
        split_counts={k:len(v) for k,v in split_rows.items()}, original_points=sum(r['original_points'] for r in rows),
        cpu_gt_feature_independence=comparisons,
        limitations=['GT independence tested through CPU network inputs only; predictions require checkpoint',
                      'Author CPU/GPU boundary semantics differ; GPU candidate requires explicit validation'])
    save(out/'preflight.json', result)
    return rows


def audit_gpu(data, out):
    import time
    import torch, torch_scatter
    source_check()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=json.loads((out/'mini_frames.json').read_text())
    rows=[r for r in rows if r['scene']=='scene-0061']
    smoke=torch_scatter.scatter_mean(torch.tensor([1.,3.,8.],device='cuda'),torch.tensor([0,0,1],device='cuda')).cpu().tolist()
    result=dict(status='partial',inference_frames=0,checkpoint_loaded=False,torch=torch.__version__,torch_scatter=torch_scatter.__version__,gpu=torch.cuda.get_device_name(),scatter_smoke=smoke,frames=[])
    for i in [0,19,38]:
     r=rows[i]; cpu=processor([r],data);gpu=processor([r],data,True)
     a,pa=cpu.process_batch_cpu(0)
     pc,labels=gpu.load_batch_to_gpu(0);b,pb=gpu.process_batch_gpu(pc,labels)
     bp=pb.cpu().numpy();ap=a['voxels'].numpy();bv=b['voxels'].cpu().numpy()
     n=r['original_points']
     repeated=[]
     for _ in range(3):
      pc,labels=gpu.load_batch_to_gpu(0);br,pr=gpu.process_batch_gpu(pc,labels)
      repeated.append(bool(torch.equal(b['voxels'],br['voxels']) and torch.equal(b['coors'],br['coors'])))
     variants=[]
     for mode in ['actual','permuted']:
      gp=processor([r],data,True,mode);pc,l=gp.load_batch_to_gpu(0);bc,pp=gp.process_batch_gpu(pc,l)
      variants.append(bool(torch.equal(b['voxels'],bc['voxels']) and torch.equal(b['coors'],bc['coors'])))
     result['frames'].append(dict(sample_token=r['sample_token'],original_points=n,cpu_processed_points=len(pa),gpu_processed_points=len(bp),
       original_prefix_equal=bool(np.array_equal(pa[:n],bp[:n])),all_features_equal=bool(np.array_equal(ap,bv)),
       original_projection_disagreements=int((a['coors'][:n].numpy()!=b['coors'][:n].cpu().numpy()).any(1).sum()),
       gpu_repeated_exact=repeated,gpu_gt_feature_independence=variants))
    # Profile actual same raw input, warmup then 30 interleaved paired preprocessing rounds.
    cp=processor([rows[0]],data);gp=processor([rows[0]],data,True)
    for _ in range(3):
     cp.process_batch_cpu(0);pc,l=gp.load_batch_to_gpu(0);gp.process_batch_gpu(pc,l)
    times={'cpu':[],'gpu':[]}
    for i in range(30):
     for name in (['cpu','gpu'] if i%2==0 else ['gpu','cpu']):
      torch.cuda.synchronize();t=time.perf_counter()
      if name=='cpu':
       batch,pc=cp.process_batch_cpu(0);v=batch['voxels'].cuda();c=batch['coors'].cuda();v.cpu();c.cpu()
      else:
       pc,l=gp.load_batch_to_gpu(0);batch,pc=gp.process_batch_gpu(pc,l);batch['voxels'].cpu();batch['coors'].cpu()
      torch.cuda.synchronize();times[name].append((time.perf_counter()-t)*1000)
    result['paired_preprocessing']={k:dict(rounds=30,p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),mean_ms=float(np.mean(v)),std_ms=float(np.std(v)),raw_ms=v) for k,v in times.items()}
    result['timing_boundary']='raw file read + preprocessing + H2D + voxel/coor D2H, synchronized; no model/GT read/visualization; warm cache'
    result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    result['adoption']='not adopted: numerical/semantic equivalence and model mIoU not validated'
    save(out/'gpu_preprocessing_audit.json',result)
    return result


def checkpoint_net(checkpoint):
    """Only unwrap the uniform DDP prefix explicitly supported by author manager."""
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('net'), dict):
        raise ValueError("Author checkpoint must contain complete checkpoint['net']")
    state = checkpoint['net']
    prefixed = [key.startswith('module.') for key in state]
    if not state or (any(prefixed) and not all(prefixed)):
        raise ValueError('Empty or mixed-prefix checkpoint state')
    return ({k[7:]: v for k, v in state.items()} if all(prefixed) else state), bool(all(prefixed))


def run_inference(data, out, checkpoint_path):
    """Actual author network; never invoked unless full checkpoint exists."""
    import time
    import resource
    import torch
    import yaml
    source_check()
    if not torch.cuda.is_available():
        raise RuntimeError('RTX4090 CUDA execution required for this experiment')
    torch.set_num_threads(2)
    torch.manual_seed(20261013)
    np.random.seed(20261013)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    from core.network import Network
    cfg = yaml.safe_load((SOURCE/'configs/net/harpnext-nuscenes.yaml').read_text())
    assert cfg['range_proj'] == dict(range_H=32, range_W=480, fov_up=10., fov_down=-30.)
    cold_start = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state, prefix_removed = checkpoint_net(checkpoint)
    model = Network('harpnext', cfg).build_network()
    incompatible = model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    torch.cuda.synchronize()
    load = dict(status='passed', strict=True, missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys), uniform_module_prefix_removed=prefix_removed,
        state_tensors=len(state), checkpoint_sha256=sha256(checkpoint_path),
        parameter_count=sum(p.numel() for p in model.parameters()),
        cold_start_load_seconds=time.perf_counter()-cold_start)
    save(out/'strict_load.json', load)
    hooks, evidence = [], {}
    def shapes(value):
        if isinstance(value, torch.Tensor):
            return {'shape':list(value.shape), 'dtype':str(value.dtype)}
        if isinstance(value, dict):
            return {str(k):shapes(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)):
            return [shapes(v) for v in value]
        return type(value).__name__
    def hook(name):
        def record(module, inputs, output):
            item = evidence.setdefault(name, dict(class_name=type(module).__name__, calls=0))
            item['calls'] += 1
            item['input'] = shapes(inputs)
            item['output'] = shapes(output)
        return record
    required = ['voxel_encoder', 'backbone', 'backbone.fusion_stem', 'decode_head.conv_seg']
    conv_names = []
    for name, module in model.named_modules():
        if type(module).__name__ == 'ConvSENeXt':
            conv_names.append(name)
        if name in required or type(module).__name__ == 'ConvSENeXt' or 'point_fusion_layers.' in name and name.count('.') == 2:
            hooks.append(module.register_forward_hook(hook(name)))
    if not conv_names:
        raise RuntimeError('Expected author ConvSENeXt modules missing')
    rows = json.loads((out/'mini_frames.json').read_text())
    scenes = [r for r in rows if r['scene']=='scene-0061']
    val = [r for r in rows if r['scene'] in ['scene-0103','scene-0916']]
    local = ROOT/'runs/perception_2025_20261013/harpnext'
    local.mkdir(parents=True, exist_ok=True)
    def predict(row, mode='zero'):
        torch.cuda.synchronize()
        start = time.perf_counter()
        proc = processor([row], data, labels_mode=mode)
        batch, points = proc.process_batch_cpu(0)
        n = row['original_points']
        original = np.fromfile(data/row['point_path'], dtype=np.float32).reshape(-1,5)[:,:4]
        if not np.array_equal(points[:n], original):
            raise ValueError('Author preprocessing no longer preserves original prefix')
        preproc_ms = (time.perf_counter()-start)*1000
        transfer_start = time.perf_counter()
        inputs = {'points':[v.cuda(non_blocking=True) for v in batch['points']],
                  'voxels':{'voxels':batch['voxels'].cuda(non_blocking=True),
                            'coors':batch['coors'].cuda(non_blocking=True)}}
        torch.cuda.synchronize()
        h2d_ms = (time.perf_counter()-transfer_start)*1000
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            begin.record()
            logits = model(inputs, training=False)['seg_logits']
            end.record()
            end.synchronize()
            if logits.shape != (len(points),16) or not torch.isfinite(logits).all():
                raise ValueError('Nonfinite or misaligned per-point logits')
            d2h_start = time.perf_counter()
            all_pred = logits.argmax(1).cpu().numpy()
            original_logits = logits[:n].cpu().numpy()
        mapping = np.concatenate([np.arange(n), np.full(len(points)-n,-1)])
        pred = original_predictions(all_pred, mapping, n)
        return pred, original_logits, dict(original_points=n, appended_points=len(points)-n,
            preproc_ms=preproc_ms, h2d_ms=h2d_ms, gpu_forward_ms=begin.elapsed_time(end),
            decode_d2h_ms=(time.perf_counter()-d2h_start)*1000,
            end_to_end_ms=(time.perf_counter()-start)*1000)
    # Gate one frame, then first/middle/last before any expanded evaluation.
    gate = []
    for row in [scenes[0], scenes[len(scenes)//2], scenes[-1]]:
        pred, logits, _ = predict(row)
        comparisons = []
        for mode in ['zero','actual','permuted']:
            other, other_logits, _ = predict(row, mode)
            comparisons.append(dict(labels_mode=mode, predictions_equal=bool(np.array_equal(pred,other)),
                logits_max_abs=float(np.max(np.abs(logits-other_logits)))))
        gate.append(dict(sample_token=row['sample_token'], comparisons=comparisons))
    save(out/'gt_prediction_independence.json',gate)
    if any(not c['predictions_equal'] for frame in gate for c in frame['comparisons']):
        raise RuntimeError('Network prediction repeat/GT independence gate failed; evaluate cause before expansion')
    missing = [n for n in required + conv_names if evidence.get(n,{}).get('calls',0)==0]
    save(out/'module_hooks.json',dict(modules=evidence, zero_call_required=missing))
    if missing:
        raise RuntimeError('Critical author module hook not called')
    for handle in hooks:
        handle.remove()
    histograms = {k:np.zeros((16,16),dtype=np.int64) for k in ['scene0061','mini_val','mini_all']}
    measured = []
    seen = set()
    for row in scenes + val + rows:
        if row['sample_token'] in seen:
            continue
        seen.add(row['sample_token'])
        pred, _, timing = predict(row)
        # The real ground truth is loaded after prediction and never supplied to default inference.
        gt = map_labels(np.fromfile(data/row['label_path'],dtype=np.uint8))
        hist = confusion(gt,pred)
        histograms['mini_all'] += hist
        if row['scene']=='scene-0061':
            histograms['scene0061'] += hist
        if row['scene'] in ['scene-0103','scene-0916']:
            histograms['mini_val'] += hist
        destination = local/(row['sample_token']+'.npz')
        np.savez_compressed(destination,prediction=pred.astype(np.uint8),original_count=len(pred))
        measured.append(dict(sample_token=row['sample_token'],sample_data_token=row['sample_data_token'],
            scene=row['scene'], prediction_path=str(destination), prediction_sha256=sha256(destination),**timing))
        save(out/'inference_progress.json',dict(status='running',inference_frames=len(measured),total=404))
    metrics = {key:dict(**summarize(hist),frame_count=39 if key=='scene0061' else 81 if key=='mini_val' else 404,
                       source='new_measured',scope='nuScenes mini teaching diagnostic; pretraining overlap possible')
               for key,hist in histograms.items()}
    save(out/'metrics.json',metrics)
    save(out/'inference_frames.json',measured)
    save(out/'profile.json',dict(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
         peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         dtype='float32',tf32=False,batch_size=1,
         boundary='CPU processor construction/raw read/preprocess + H2D + CUDA forward + point logits/prediction D2H; no evaluation or saving'))
    result = dict(method='HARP-NeXt',status='partial',inference_frames=len(measured),strict_load=load,
        metrics='metrics.json',preprocessing='native CPU; GPU candidate not adopted',
        remaining=['GPU equivalent preprocessing and paired model end-to-end timing','synchronized replay integration'])
    save(out/'status.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['preflight','run','optimize','audit_gpu'], default='preflight')
    p.add_argument('--config', type=Path)
    p.add_argument('--data', type=Path, default=ROOT/'data/nuscenes')
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/harpnext_2025/ckpt_best.pth')
    args = p.parse_args()
    if args.config:
        import yaml
        config = yaml.safe_load(args.config.read_text())
        method = config.get('models', {}).get('harpnext', {})
        for key, expected in [('source', SOURCE), ('config', SOURCE/'configs/net/harpnext-nuscenes.yaml')]:
            base = SOURCE if key == 'config' else ROOT
            if key in method and (base/method[key]).resolve() != expected.resolve():
                raise ValueError('Unsupported HARP-NeXt config override: '+key)
        if 'data' in config:
            args.data = ROOT/config['data']
        if 'report' in config:
            args.out = ROOT/config['report']/'harpnext'
        if method.get('checkpoint'):
            args.checkpoint = ROOT/method['checkpoint']
    if args.phase == 'audit_gpu':
        audit_gpu(args.data, args.out)
        return 0
    preflight(args.data, args.out)
    if args.phase != 'preflight':
        if not args.checkpoint.is_file():
            result = dict(method='HARP-NeXt', status='blocked_resource', phase=args.phase,
                reason='Matching complete author nuScenes32x480 checkpoint not downloaded; see resource_manifest.json',
                inference_frames=0, metrics=None, strict_load=None, gpu_ab=None,
                source_commit=COMMIT, checkpoint_path=str(args.checkpoint),
                implementation_scope='Author model adapter implemented but untested without complete checkpoint',
                preprocessing_audit='partial; gpu_preprocessing_audit.json',
                original_point_audit='passed;404frames/14026208points',
                adoption='pending strict checkpoint load and genuine model execution')
            save(args.out/'status.json', result)
            print(json.dumps(result))
            return 2
        if args.phase == 'optimize':
            result = audit_gpu(args.data, args.out)
            save(args.out/'optimization.json', result)
            return 0
        try:
            run_inference(args.data, args.out, args.checkpoint)
        except Exception as exc:
            save(args.out/'status.json', dict(method='HARP-NeXt', status='failed',
                reason=type(exc).__name__ + ': ' + str(exc), phase=args.phase,
                metrics=None, inference_frames=0, partial_outputs='Inspect inference_progress.json when present'))
            raise
        return 0
    print(json.dumps({'status':'passed', 'phase':'preflight'}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
