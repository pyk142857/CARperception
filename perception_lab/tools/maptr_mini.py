"""Run official camera-only MapTR weights on mini, without loading map annotations."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from evidence import sha256
from maptr_geometry import CAMERAS, CLASSES, lidar_to_image, vectors_to_ego, validate_frames

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/maptr')
    args = parser.parse_args()
    args.out = args.out.resolve()
    import torch
    import mmcv
    from pyquaternion import Quaternion
    torch.set_num_threads(2)
    torch.manual_seed(20260924)
    np.random.seed(20260924)
    repo = ROOT/'third_party/MapTR'
    sys.path[:0] = [str(repo), str(repo/'mmdetection3d')]
    os.chdir(repo)
    import projects.mmdet3d_plugin  # register official modules
    from mmdet3d.models import build_model
    from mmdet3d.datasets.pipelines import Compose
    config = repo/'projects/configs/maptr/maptr_tiny_r50_24e.py'
    checkpoint = ROOT/'checkpoints/maptr_tiny_r50_24e.pth'
    cfg = mmcv.Config.fromfile(str(config))
    cfg.model.pretrained = None  # all parameters come from the complete trained checkpoint
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    saved = torch.load(checkpoint, map_location='cpu')
    weights = saved.get('state_dict', saved)
    weights = {k.removeprefix('module.'):v for k,v in weights.items()}
    # This official checkpoint stores a deterministic grid buffer which newer
    # upstream code marks non-persistent. Verify its value, then load it strictly.
    for name, module in model.named_modules():
        key = name + '.grid_offsets'
        if key in weights and 'grid_offsets' in module._non_persistent_buffers_set:
            if not torch.equal(module.grid_offsets.cpu(), weights[key]):
                raise ValueError('Checkpoint grid offsets differ from configured kernel')
            module.register_buffer('grid_offsets', module.grid_offsets, persistent=True)
    model.load_state_dict(weights, strict=True)
    model.cuda().eval()
    print('Strict checkpoint load passed', flush=True)
    pipeline = Compose(cfg.test_pipeline)
    packets = json.loads((ROOT/'manifests/frame_packets.json').read_text())
    if args.limit:
        packets = packets[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, packet in enumerate(packets):
        lidar = packet['sensors']['LIDAR_TOP']
        pose = np.asarray(lidar['T_ego_to_global'])
        q = Quaternion(matrix=pose[:3,:3])
        yaw = np.arctan2(pose[1,0], pose[0,0]) % (2*np.pi)
        # Official converter uses zeros when optional CAN telemetry is absent.
        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[:3] = pose[:3,3]
        can_bus[3:7] = q.elements
        can_bus[-2:] = [yaw, np.degrees(yaw)]
        inputs = dict(img_filename=[packet['sensors'][c]['path'] for c in CAMERAS],
                      lidar2img=[lidar_to_image(lidar,packet['sensors'][c]) for c in CAMERAS],
                      sample_idx=packet['sample_token'], scene_token=packet.get('scene_token','scene-0061'),
                      can_bus=can_bus, img_fields=[], bbox3d_fields=[], pts_mask_fields=[],
                      pts_seg_fields=[], bbox_fields=[], mask_fields=[], seg_fields=[])
        data = pipeline(inputs)
        image = data['img'][0].data.unsqueeze(0).cuda()
        meta = data['img_metas'][0].data
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            pred = model(return_loss=False, rescale=True, img=[image], img_metas=[[meta]])[0]['pts_bbox']
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        xy = pred['pts_3d'].numpy()
        scores = pred['scores_3d'].numpy()
        labels = pred['labels_3d'].numpy()
        xyz = vectors_to_ego(xy, lidar['T_sensor_to_ego'])
        vectors = [dict(class_name=CLASSES[int(label)], score=float(score),
                        points_lidar_xy=line.tolist(), points_xyz=ego.tolist())
                   for label, score, line, ego in zip(labels,scores,xy,xyz)]
        results.append(dict(sample_token=packet['sample_token'], timestamp_us=packet['timestamp_us'],
                            source='measured', status='passed', model='MapTR-tiny-R50-24e',
                            coordinate_system='reference_ego_FLU', display_ground_z=0.,
                            inference_seconds=elapsed, vectors=vectors))
        print(f'frame {i+1}/{len(packets)}: {sum(scores>=.5)} vectors >= 0.5, {elapsed:.3f}s', flush=True)
    validate_frames(results, packets)
    output = args.out/'predictions.json'
    output.write_text(json.dumps(results, indent=2, allow_nan=False))
    sources = [checkpoint, config, Path(__file__), ROOT/'tools/maptr_geometry.py', ROOT/'manifests/frame_packets.json']
    summary = dict(status='passed', source='measured', frames=len(results), strict_checkpoint=True,
                   model='MapTR-tiny-R50-24e', camera_order=CAMERAS, map_ground_truth_used=False,
                   can_bus='pose/quaternion/yaw from metadata; missing acceleration/rotation-rate/velocity set to zero (upstream fallback)',
                   geometry='model predicts LiDAR XY; calibrated XY transformed to ego; display Z=0 is approximate, not predicted',
                   source_sha256={str(p):sha256(p) for p in sources},
                   artifacts={str(output.resolve()):sha256(output)},
                   torch_version=torch.__version__, mmcv_version=mmcv.__version__,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Saved {output}', flush=True)

if __name__ == '__main__':
    main()
