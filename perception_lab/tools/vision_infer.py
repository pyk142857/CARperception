"""Official-model diagnostic / mini inference, with native and adapted outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from evidence import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['yolo', 'depth'], required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.device == 'cpu':
        # Upstream depth image2tensor auto-selects CUDA when visible.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    if args.model == 'yolo':
        from ultralytics import YOLO
        weight = root / 'checkpoints/yolov8s.pt'
        model = YOLO(weight)
        module = 'M01'
    else:
        sys.path.insert(0, str(root / 'third_party/depth_anything_v2/metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
        weight = root / 'checkpoints/depth_anything_v2_metric_vkitti_vits.pth'
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=80)
        state = torch.load(weight, map_location='cpu')
        print(model.load_state_dict(state, strict=True), flush=True)
        model = model.to(args.device).eval()
        module = 'M03'
    digest = sha256(weight)
    config = {'model': args.model, 'device': args.device, 'weight_sha256': digest,
              'scope': manifest['scope'], 'source': 'measured', 'torch': torch.__version__,
              'preprocess': 'official predict imgsz=640 conf=0.25 iou=0.7' if module == 'M01' else 'official infer_image input_size=518 max_depth=80'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    predictions = []
    start = time.perf_counter()
    for index, frame in enumerate(manifest['images']):
        image = cv2.imread(frame['path'])
        if image is None:
            raise ValueError('Unreadable image: ' + frame['path'])
        prefix = f'{index:06d}'
        bundle = dict(run_id=out.parts[-3], module_id=module, model_id=args.model,
                      sample_token=frame.get('sample_token'), scene_token=frame.get('scene_token'),
                      timestamp_us=frame.get('timestamp_us'), status='passed',
                      scope=manifest['scope'], source='measured', input_path=frame['path'],
                      input_sha256=sha256(frame['path']), checkpoint_sha256=digest,
                      source_config_hash=sha256(out / 'config.json'))
        with torch.inference_mode():
            if module == 'M01':
                r = model.predict(image, imgsz=640, conf=0.25, iou=0.7, device=args.device, verbose=False)[0]
                native = r.boxes.data.cpu().numpy()
                np.save(out / (prefix + '_native.npy'), native)
                (out / 'labels.json').write_text(json.dumps(r.names, indent=2))
                bundle.update(native_coordinate_system='original_image_xyxy', label_namespace='COCO80',
                              boxes2d=[dict(xyxy=b[:4].tolist(), score=float(b[4]), class_id=int(b[5]),
                                            camera_channel=frame.get('camera_channel')) for b in native])
                if len(bundle['boxes2d']) != len(native) or not np.isfinite(native).all():
                    raise ValueError('Invalid detection adapter output')
                cv2.imwrite(str(out / (prefix + '_overlay.jpg')), r.plot())
            else:
                depth = model.infer_image(image).astype(np.float32)
                if depth.shape != image.shape[:2] or not np.isfinite(depth).all() or not (depth > 0).all():
                    raise ValueError('Invalid metric depth output')
                np.save(out / (prefix + '_depth_m.npy'), depth)
                np.save(out / (prefix + '_valid_mask.npy'), np.ones_like(depth, dtype=bool))
                bundle.update(native_coordinate_system='camera_Z_m', label_namespace='metric_depth_m',
                              depth={'path':str(out / (prefix + '_depth_m.npy')), 'unit':'m',
                                     'shape': list(depth.shape), 'min':float(depth.min()), 'max':float(depth.max())})
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(12, 5))
                handle = ax.imshow(depth, vmin=0, vmax=80, cmap='turbo'); ax.axis('off')
                fig.colorbar(handle, ax=ax, label='Camera Z depth (m)')
                fig.savefig(out / (prefix + '_depth.png'), bbox_inches='tight'); plt.close(fig)
        predictions.append(bundle)
        print(prefix, frame['path'], 'passed', flush=True)
    (out / 'predictions.json').write_text(json.dumps(predictions, indent=2))
    (out / 'summary.json').write_text(json.dumps({'source':'measured', 'scope':manifest['scope'],
        'sample_count':len(predictions), 'device':args.device, 'wall_seconds':time.perf_counter()-start,
        'timing_scope':'diagnostic inference and artifact writing; not a performance benchmark',
        'metric_value':None,'metric_reason':'No task evaluation performed by this command'}, indent=2))


if __name__ == '__main__':
    main()
