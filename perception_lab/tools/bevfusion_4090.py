"""Pinned NVIDIA R50 BEVFusion inference on fixed raw nuScenes packets.

No annotation enters preprocessing or forward. The official deterministic test
pipeline is retained with annotation/map-target loaders removed.
"""
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = 'db75150717a9462cb60241e36ba28d65f6908607'
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def packet_input(packet):
    """Column-vector transforms including each asynchronous sensor ego pose."""
    sensors = packet['sensors']
    lidar = sensors['LIDAR_TOP']
    l2e = np.asarray(lidar['T_sensor_to_ego'], dtype=np.float64)
    e2g = np.asarray(lidar['T_ego_to_global'], dtype=np.float64)
    g2l = np.linalg.inv(e2g @ l2e)
    data = dict(token=packet['sample_token'], sample_idx=packet['sample_token'],
                timestamp=lidar['timestamp_us'], lidar_path=lidar['path'],
                lidar2ego=l2e.astype(np.float32), ego2global=e2g.astype(np.float32),
                image_paths=[], lidar2camera=[], lidar2image=[], camera2ego=[],
                camera_intrinsics=[], camera2lidar=[], sweeps=[],
                img_fields=[], bbox3d_fields=[], pts_mask_fields=[],
                pts_seg_fields=[], bbox_fields=[], mask_fields=[], seg_fields=[])
    for name in CAMERAS:
        cam = sensors[name]
        c2e = np.asarray(cam['T_sensor_to_ego'])
        c2l = g2l @ np.asarray(cam['T_ego_to_global']) @ c2e
        l2c = np.linalg.inv(c2l)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = cam['K']
        data['image_paths'].append(cam['path'])
        for key, value in [('camera2ego', c2e), ('camera2lidar', c2l),
                           ('lidar2camera', l2c), ('lidar2image', intrinsic @ l2c),
                           ('camera_intrinsics', intrinsic)]:
            data[key].append(value.astype(np.float32))
    for sweep in packet['lidar_sweeps'][:9]:
        s2l = g2l @ np.asarray(sweep['T_ego_to_global']) @ np.asarray(sweep['T_sensor_to_ego'])
        data['sweeps'].append(dict(data_path=sweep['path'], timestamp=sweep['timestamp_us'],
                                  sensor2lidar_rotation=s2l[:3, :3], sensor2lidar_translation=s2l[:3, 3]))
    return data


def standardize(detection, packet, classes):
    """Reuse pinned upstream wlh/yaw convention, then rigidly transform to ego."""
    from mmdet3d.datasets.nuscenes_dataset import output_to_nusc_box
    from pyquaternion import Quaternion
    native = output_to_nusc_box(detection)
    matrix = np.asarray(packet['sensors']['LIDAR_TOP']['T_sensor_to_ego'])
    q = Quaternion(matrix=matrix[:3, :3])
    records, audit = [], []
    for b in native:
        center, rotation, velocity = b.center.copy(), b.orientation, np.asarray(b.velocity).copy()
        b.rotate(q)
        b.translate(matrix[:3, 3])
        records.append(dict(center_xyz=b.center.tolist(), size_wlh=b.wlh.tolist(),
                            rotation_wxyz=b.orientation.elements.tolist(), velocity_xy=np.asarray(b.velocity)[:2].tolist(),
                            class_name=classes[b.label], score=float(b.score)))
        audit.append(dict(center=float(np.max(np.abs(matrix[:3, :3].T @ (b.center - matrix[:3, 3]) - center))),
                          rotation=float(np.max(np.abs((q.inverse * b.orientation).rotation_matrix - rotation.rotation_matrix))),
                          velocity=float(np.max(np.abs(matrix[:3, :3].T @ b.velocity - velocity)))))
    if audit and max(max(a.values()) for a in audit) > 1e-7:
        raise ValueError('Native-to-ego roundtrip failed')
    return records, dict(sample_token=packet['sample_token'], boxes=len(native),
                         max_error={k: max((a[k] for a in audit), default=0.) for k in ['center', 'rotation', 'velocity']})


def make_pipeline(cfg):
    from mmdet3d.datasets.pipelines import Compose
    steps = copy.deepcopy(cfg.test_pipeline)
    steps = [s for s in steps if s['type'] not in ['LoadAnnotations3D', 'LoadBEVSegmentation']]
    for step in steps:
        if step['type'] == 'LoadPointsFromMultiSweeps':
            step['test_mode'] = True
        if step['type'] == 'DefaultFormatBundle3D':
            step['with_gt'] = False
        if step['type'] == 'Collect3D':
            step['keys'] = [k for k in step['keys'] if not k.startswith('gt_')]
    return Compose(steps), steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'model-config', 'checkpoint', 'dataroot', 'packets', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--capture-pooling', type=Path)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    identity = {key: digest(getattr(a, key)) for key in ['model_config', 'checkpoint', 'packets']}
    config_files = [a.model_config.resolve()]
    config_files += [parent / 'default.yaml' for parent in a.model_config.resolve().parents if (parent / 'default.yaml').exists()]
    identity['config_hierarchy'] = {str(path): digest(path) for path in config_files}
    identity['config_sha256'] = hashlib.sha256(json.dumps(identity['config_hierarchy'], sort_keys=True).encode()).hexdigest()
    identity['compatibility_patch_sha256'] = digest(Path(__file__).with_name('bevfusion_torch113_sm89.patch'))
    raw_packets = json.loads(a.packets.read_text())
    selected_packets = raw_packets[:a.limit] if a.limit is not None else raw_packets
    sensor_paths = {sensor['path'] for packet in selected_packets for sensor in list(packet['sensors'].values()) + packet['lidar_sweeps'][:9]}
    identity['sensor_files'] = {path: digest(path) for path in sorted(sensor_paths)}
    identity.update(upstream_commit=UPSTREAM_COMMIT, adapter_sha256=digest(__file__), limit=a.limit,
                    precision='fp32', camera_order=CAMERAS)
    manifest_path = a.out / 'manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if prior.get('identity') == identity and prior.get('status') == 'passed':
            if all(digest(a.out / name) == sha for name, sha in prior['outputs_sha256'].items()):
                print('Reusing verified matching measured artifacts:', a.out)
                return
    start = time.perf_counter()
    sys.path.insert(0, str(a.source.resolve()))
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    from mmdet3d.models import build_model
    import subprocess
    actual_commit = subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise ValueError('Wrong BEVFusion commit: ' + actual_commit)
    configs.load(str(a.model_config.resolve()), recursive=True)
    cfg = Config(recursive_eval(configs), filename=str(a.model_config))
    if cfg.model.encoders.camera.backbone.get('depth') != 50:
        raise ValueError('This stage requires the NVIDIA ResNet50 model')
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.encoders.camera.backbone.init_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(a.checkpoint, map_location='cpu')
    weights = checkpoint.get('state_dict', checkpoint)
    if all(k.startswith('module.') for k in weights):
        weights = {k[7:]: v for k, v in weights.items()}
    incompatible = model.load_state_dict(weights, strict=True)
    write_json(a.out / 'strict_load.json', dict(missing=list(incompatible.missing_keys),
                unexpected=list(incompatible.unexpected_keys), parameters=len(weights), sparse_layout='upstream unchanged'))
    del weights, checkpoint
    model = model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    pipeline, steps = make_pipeline(cfg)
    from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
    def prepare(packet):
        data = packet_input(packet)
        data.update(box_type_3d=LiDARInstance3DBoxes, box_mode_3d=Box3DMode.LIDAR)
        return pipeline(data)
    write_json(a.out / 'resolved_config.json', dict(model=cfg.model, pipeline=steps, object_classes=cfg.object_classes))
    packets = json.loads(a.packets.read_text())
    if a.limit is not None:
        packets = packets[:a.limit]
    if not packets or len({p['sample_token'] for p in packets}) != len(packets):
        raise ValueError('Packets must have unique nonempty coverage')
    executed = dict(camera=0, lidar=0, fusion=0)
    handles = []
    def hook(name):
        def count(*unused):
            executed[name] += 1
        return count
    for name, module in [('camera', model.encoders['camera']['backbone']),
                         ('lidar', model.encoders['lidar']['backbone']), ('fusion', model.fuser)]:
        handles.append(module.register_forward_hook(hook(name)))
    capture_state = {'path': None}
    if a.capture_pooling:
        import importlib
        base = importlib.import_module('mmdet3d.models.vtransforms.base')
        reference_pool = base.bev_pool
        def capture(features, coords, B, D, H, W):
            output = reference_pool(features, coords, B, D, H, W)
            if capture_state['path'] is not None:
                path = capture_state['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(features=features.detach().cpu(), coords=coords.detach().cpu(),
                                output=output.detach().cpu(), B=B, D=D, H=H, W=W), path)
                capture_state['path'] = None
            return output
        base.bev_pool = capture
    cold = time.perf_counter() - start
    predictions, native_records, audits, times, inputs = [], [], [], [], []
    with torch.inference_mode():
        first = scatter(collate([prepare(packets[0])], samples_per_gpu=1), [0])[0]
        for _ in range(a.warmup):
            model(**first)
        torch.cuda.synchronize()
        del first
        torch.cuda.reset_peak_memory_stats()
        for index, packet in enumerate(packets):
            t0 = time.perf_counter()
            prepared = prepare(packet)
            point_count = len(prepared['points'].data)
            batch = collate([prepared], samples_per_gpu=1)
            t1 = time.perf_counter()
            batch = scatter(batch, [0])[0]
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = model(**batch)[0]
            end.record()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            result = {k: v.to('cpu') if hasattr(v, 'to') else v for k, v in result.items()}
            boxes, audit = standardize(result, packet, cfg.object_classes)
            rec = {k: packet[k] for k in ['sample_token', 'scene_token', 'timestamp_us']}
            rec.update(source='measured', model='bevfusion_reference', boxes3d=boxes,
                       checkpoint_sha256=identity['checkpoint'], source_config_hash=identity['config_sha256'])
            predictions.append(rec)
            native_records.append(dict(sample_token=packet['sample_token'],
                tensor=result['boxes_3d'].tensor.tolist(), scores=result['scores_3d'].tolist(), labels=result['labels_3d'].tolist()))
            audits.append(audit)
            t4 = time.perf_counter()
            times.append(dict(frame=index, sample_token=packet['sample_token'], points=point_count,
                preprocess_ms=(t1-t0)*1000, h2d_ms=(t2-t1)*1000, forward_sync_ms=(t3-t2)*1000,
                forward_cuda_event_ms=begin.elapsed_time(end), d2h_standardize_ms=(t4-t3)*1000, end_to_end_ms=(t4-t0)*1000))
            inputs.append(dict(sample_token=packet['sample_token'], points=point_count, image_shape=list(batch['img'].shape),
                sweeps=[s['sample_data_token'] for s in packet['lidar_sweeps'][:9]],
                first_frame_pad_empty=not packet['lidar_sweeps']))
            if a.capture_pooling and index in {0, len(packets) // 2, len(packets) - 1}:
                capture_state['path'] = a.capture_pooling / ('pool_capture_%d.pt' % index)
                model(**batch)
                torch.cuda.synchronize()
            write_json(a.out / 'predictions.partial.json', predictions)
            print(json.dumps(times[-1]), flush=True)
    for handle in handles:
        handle.remove()
    write_json(a.out / 'predictions.json', predictions)
    write_json(a.out / 'native_predictions.json', native_records)
    write_json(a.out / 'coordinate_audit.json', dict(convention='pinned output_to_nusc_box; dims=wlh, yaw=-native_yaw-pi/2; gravity center then lidar-to-ego', frames=audits))
    write_json(a.out / 'actual_inputs.json', inputs)
    with (a.out / 'timings.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(times[0])); writer.writeheader(); writer.writerows(times)
    summary = {}
    for key in ['preprocess_ms', 'h2d_ms', 'forward_sync_ms', 'forward_cuda_event_ms', 'd2h_standardize_ms', 'end_to_end_ms']:
        values = [t[key] for t in times]
        summary[key] = dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)), first=values[0])
    write_json(a.out / 'performance.json', dict(frames=len(times), cold_start_seconds=cold, warmup=a.warmup,
        allocated_peak_bytes=torch.cuda.max_memory_allocated(), reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        branch_executions=executed, timings=summary, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, boundaries='forward includes upstream output.cpu transfers; preprocessing includes raw sensor IO; JSON disk writes excluded'))
    outputs = ['predictions.json', 'native_predictions.json', 'coordinate_audit.json', 'actual_inputs.json', 'timings.csv', 'performance.json', 'resolved_config.json', 'strict_load.json']
    write_json(manifest_path, dict(status='passed', identity=identity, frames=len(times),
        outputs_sha256={name: digest(a.out / name) for name in outputs}, official_eval_status='not_run'))


if __name__ == '__main__':
    main()
