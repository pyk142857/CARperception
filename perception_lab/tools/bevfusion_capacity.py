"""Synthetic capacity-boundary checks; these are not detection-accuracy runs."""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from bevfusion_4090 import digest, write_json, packet_input, make_pipeline

ROOT = Path(__file__).resolve().parents[1]


def verify_runner_rejection(capacity):
    """Execute the actual runner's unchanged point-count guard AST on oversized input."""
    path = ROOT / 'tools/bevfusion_trt_4090.py'
    source = path.read_text()
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and 'Point capacity exceeded' in ast.get_source_segment(source, node)]
    if len(matches) != 1:
        raise AssertionError('Could not uniquely identify actual runner capacity guard')
    code = compile(ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[])), str(path), 'exec')
    points = np.empty((capacity + 1, 5), dtype=np.float16)
    try:
        exec(code, {'points': points, 'tcfg': {'max_input_points': capacity}})
    except ValueError as error:
        return dict(status='passed', input_points=len(points), rejected_before_backend=True,
                    error=str(error), runner_sha256=digest(path), guard=ast.get_source_segment(source, matches[0]))
    raise AssertionError('Actual runner guard did not reject over-capacity input')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/deployment_4090.yaml')
    p.add_argument('--packets', type=Path, required=True)
    p.add_argument('--case', choices=['max', 'min', 'reject'], required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    deploy = json.loads(a.config.read_text())
    capacity = deploy['tensorrt']['max_input_points']
    if a.case == 'reject':
        result = verify_runner_rejection(capacity)
        write_json(a.out, result); print(json.dumps(result)); return
    source = ROOT / deploy['bevfusion']['source']
    build = ROOT / deploy['tensorrt']['source'] / 'CUDA-BEVFusion/build'
    model = ROOT / deploy['tensorrt']['model_dir']
    sys.path[:0] = [str(source), str(build)]
    import torch
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
    import libpybev
    configs.load(str(ROOT / deploy['bevfusion']['config']), recursive=True)
    cfg = Config(recursive_eval(configs)); pipeline, _ = make_pipeline(cfg)
    packet = json.loads(a.packets.read_text())[0]
    item = packet_input(packet)
    item.update(box_type_3d=LiDARInstance3DBoxes, box_mode_3d=Box3DMode.LIDAR)
    item = pipeline(item)
    arrays = {key: np.ascontiguousarray(item[key].data.numpy(), dtype=np.float32)
              for key in ['img', 'points', 'camera2lidar', 'camera_intrinsics', 'lidar2image', 'img_aug_matrix']}
    original = arrays['points']
    count = capacity if a.case == 'max' else 1
    points = np.tile(original, ((count + len(original) - 1) // len(original), 1))[:count].astype(np.float16)
    images = arrays['img'][None].astype(np.float16)
    core = libpybev.load_bevfusion(str(model/'build/camera.backbone.plan'), str(model/'build/camera.vtransform.plan'),
        str(model/'lidar.backbone.xyz.onnx'), str(model/'build/fuser.plan'),
        str(model/'build'/(deploy['tensorrt'].get('head_engine','head.bbox')+'.plan')), 'fp16')
    if core is None:
        raise RuntimeError('Failed to construct capacity test core')
    core.update(*[arrays[key][None] for key in ['camera2lidar', 'camera_intrinsics', 'lidar2image', 'img_aug_matrix']])
    torch.cuda.synchronize(); start = time.perf_counter()
    result = np.asarray(core.forward(images, points, False, False)).copy()
    torch.cuda.synchronize(); elapsed = (time.perf_counter() - start) * 1000
    assert result.ndim == 2 and result.shape[1] == 11 and np.isfinite(result).all()
    record = dict(status='passed', scope='synthetic capacity test, not accuracy evaluation or representative model timing',
                  source_sample_token=packet['sample_token'], original_points=len(original), input_points=count,
                  preparation='tile actual preprocessed frame points then truncate' if a.case == 'max' else 'take first actual preprocessed point',
                  native_boxes=len(result), finite_output=True, one_call_wall_ms=elapsed, capacity=capacity,
                  config_sha256=digest(a.config), runner_sha256=digest(ROOT/'tools/bevfusion_trt_4090.py'),
                  core_sha256=digest(build/'libbevfusion_core.so'))
    write_json(a.out, record); print(json.dumps(record))


if __name__ == '__main__':
    main()
