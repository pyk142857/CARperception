"""Instrumented FP32 same-input repeat diagnosis; never a performance benchmark.

Run from perception_lab with envs/bevfusion/bin/python (see --help). Inputs are
preprocessed once per token. Production kernels/configuration stay unchanged.
Raw tensor files remain under runs; compact identities and metrics enter reports.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIORITY_TOKEN = '747aa46b9a4641fe90db05d97db2acea'


def select_packets(packets, cases):
    tokens = [packets[0]['sample_token'], PRIORITY_TOKEN]
    tokens += [c['sample_token'] for c in cases
               if min(c.get('reference_score', 0), c.get('candidate_score', 0)) >= .5]
    lookup = {p['sample_token']: p for p in packets}
    return [lookup[t] for t in dict.fromkeys(tokens)]


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {k: float(v) for k, v in zip(['p50', 'p95', 'p99', 'max'],
                                       np.percentile(values, [50, 95, 99, 100]))}


def tensor_comparison(reference, candidate, atol=.05, rtol=.02):
    if reference.shape != candidate.shape:
        return {'passed': False, 'reason': 'shape mismatch',
                'reference_shape': list(reference.shape), 'candidate_shape': list(candidate.shape)}
    if reference.dtype.kind in 'biu' or candidate.dtype.kind in 'biu':
        equal = reference == candidate
        return {'passed': bool(equal.all()), 'bitwise_equal': reference.dtype == candidate.dtype
                and reference.tobytes() == candidate.tobytes(), 'elements': int(reference.size),
                'nonfinite_elements': 0, 'exceeded_elements': int((~equal).sum()),
                'tolerance': 'integer identity: exact equality'}
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a[finite] - b[finite])
    magnitude = np.abs(a[finite])
    exceeded = delta > atol + rtol * magnitude
    return {'passed': bool(finite.all() and not exceeded.any()),
            'bitwise_equal': reference.dtype == candidate.dtype and reference.tobytes() == candidate.tobytes(),
            'elements': int(a.size), 'nonfinite_elements': int((~finite).sum()),
            'exceeded_elements': int(exceeded.sum()), 'absolute_error': quantiles(delta),
            'relative_error_abs_gt001': quantiles(delta[magnitude > .01] / magnitude[magnitude > .01]),
            'reference_abs_le_001_elements': int((magnitude <= .01).sum()),
            'near_zero_absolute_error': quantiles(delta[magnitude <= .01]),
            'tolerance': 'abs(candidate-reference) <= 0.05 + 0.02*abs(reference); nonfinite fails',
            'atol': atol, 'rtol': rtol}


def source_comparison(ref_sources, candidate_sources, reference, candidate):
    r = {int(s): i for i, s in enumerate(ref_sources)}
    c = {int(s): i for i, s in enumerate(candidate_sources)}
    if len(r) != len(ref_sources) or len(c) != len(candidate_sources):
        raise ValueError('Source identities must be unique class*HW+spatial_index')
    common = sorted(r.keys() & c.keys())
    delta = np.abs(reference[[r[s] for s in common]] - candidate[[c[s] for s in common]])
    return {'common_sources': len(common), 'removed_sources': sorted(r.keys() - c.keys()),
            'added_sources': sorted(c.keys() - r.keys()),
            'same_order': np.array_equal(ref_sources, candidate_sources),
            'aligned_max_abs': float(delta.max()) if delta.size else None,
            'aligned_absolute_error': quantiles(delta)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def node_layout(name, x):
    if name == 'camera.feature':
        return 'NHWC; N folds batch and six cameras'
    if name == 'camera.depth':
        return 'NDHW; N folds batch and six cameras'
    if name == 'camera.bev_pool':
        return 'BCZXY'
    if name == 'camera.pool.coords':
        return 'Nx4 integer x,y,z,batch'
    if name.startswith('camera.raster.'):
        return 'N projected points, preserving actual scatter order'
    if name.startswith('query.'):
        return 'BxK; source class*HW+spatial index' if 'pre_topk' not in name else 'Bx(class*HW)'
    if name == 'output.native_boxes':
        return 'Nx9 lidar bottom-center xyz, dimensions, yaw, vx, vy'
    if name.startswith('output.'):
        return 'N retained candidates in decode order'
    if x.ndim == 4:
        return 'NCHW'
    if name.startswith('head.') and x.ndim == 3:
        return 'BCQ'
    return 'native axes; shape recorded, no transposition'


def array_stats(x, name=''):
    finite = np.isfinite(x)
    return {'shape': list(x.shape), 'dtype': str(x.dtype), 'layout': node_layout(name, x),
            'sha256': hashlib.sha256(x.tobytes(order='C')).hexdigest(),
            'nan': int(np.isnan(x).sum()), 'inf': int(np.isinf(x).sum()),
            'abs_quantiles': quantiles(np.abs(x[finite]))}



def pooling_contributions(indices, intervals):
    intervals = np.asarray(intervals).reshape(-1, 3).astype(np.int64)
    if not len(intervals):
        return np.asarray(indices)[:0], np.empty(0, dtype=np.int64)
    counts = intervals[:, 1] - intervals[:, 0]
    if (np.any(counts < 0) or intervals[0, 0] < 0 or intervals[-1, 1] != len(indices)
            or not np.array_equal(intervals[1:, 0], intervals[:-1, 1])):
        raise ValueError('Invalid repaired exclusive pooling intervals')
    valid = np.asarray(indices)[intervals[0, 0]:intervals[-1, 1]]
    return valid, np.repeat(intervals[:, 2], counts)


def reference_replay(args, model, torch, write_json):
    """Run FP32 modules on exactly the saved FP16-representable TRT replay inputs."""
    vt = model.encoders['camera']['vtransform']
    head = model.heads['object']
    out = args.out / 'replay'
    raw_out = args.raw_out / 'replay'
    summary = {'scope': 'FP32 reference modules and production TRT modules consume identical '
                        'saved FP16-representable inputs; instrumented, no speed claim',
               'runner_sha256': sha(__file__), 'torch_version': torch.__version__, 'cases': []}
    def first(value):
        return value[0] if isinstance(value, (list, tuple)) else value
    files = sorted(args.replay_dir.glob('*/*.inputs.npz'))
    if not files:
        raise ValueError('No replay inputs found in ' + str(args.replay_dir))
    with torch.inference_mode():
        for path in files:
            token, case = path.parent.name, path.name[:-len('.inputs.npz')]
            module = case.split('.')[0]
            with np.load(path) as values:
                arrays = {k: v.copy() for k, v in values.items()}
            tensors = {k: torch.from_numpy(v.astype(np.float32)).cuda() for k, v in arrays.items()
                       if v.dtype.kind == 'f'}
            source_ids = None
            geometry_result = None
            if module == 'camera':
                img, depth = tensors['img'], tensors['depth']
                img = img.reshape(-1, *img.shape[-3:])
                x = first(model.encoders['camera']['neck'](model.encoders['camera']['backbone'](img)))
                depth = depth.reshape(-1, *depth.shape[-3:])
                x = vt.depthnet(torch.cat([vt.dtransform(depth), x], dim=1))
                results = {'camera_feature': x[:, vt.D:vt.D + vt.C].permute(0, 2, 3, 1),
                           'camera_depth_weights': x[:, :vt.D].softmax(1)}
            elif module == 'view':
                results = {'feat_out': vt.downsample(tensors['feat_in'])}
            elif module == 'pool':
                features = tensors['camera_feature']
                depths = tensors['camera_depth_weights']
                cameras, channels_depth, height, width = depths.shape
                indices = arrays['indices'].astype(np.int64)
                intervals = arrays['intervals'].reshape(-1, 3).astype(np.int64)
                indices, destinations = pooling_contributions(indices, intervals)
                coords = np.zeros((len(indices), 4), dtype=np.int32)
                coords[:, 0], coords[:, 1] = destinations // 360, destinations % 360
                idx = torch.from_numpy(indices).cuda()
                camera = idx // (channels_depth * height * width)
                spatial = idx % (height * width)
                feature_rows = features.reshape(cameras * height * width, -1)[camera * height * width + spatial]
                products = feature_rows * depths.reshape(-1)[idx, None]
                base = importlib.import_module('mmdet3d.models.vtransforms.base')
                pooled = base.bev_pool(products.contiguous(), torch.from_numpy(coords).cuda(), 1, 1, 360, 360)
                results = {'feat_in': pooled[:, :, 0]}
                # Isolate mapping only: identical products and FP32 kernel; canonical
                # ascending source order on both sides removes tie-order confounding.
                fixed = np.load(args.raw_out / token / 'preprocessed_inputs.npz')
                matrices = {k: torch.from_numpy(fixed['input.' + k]).cuda() for k in
                            ['camera2lidar', 'camera_intrinsics', 'img_aug_matrix', 'lidar_aug_matrix']}
                c2l, intr, aug, laug = [matrices[k] for k in
                                      ['camera2lidar', 'camera_intrinsics', 'img_aug_matrix', 'lidar_aug_matrix']]
                geometry = vt.get_geometry(c2l[..., :3, :3], c2l[..., :3, 3], intr[..., :3, :3],
                    aug[..., :3, :3], aug[..., :3, 3], extra_rots=laug[..., :3, :3], extra_trans=laug[..., :3, 3])
                integer_coords = ((geometry - (vt.bx - vt.dx / 2.0)) / vt.dx).long().reshape(-1, 3)
                kept = ((integer_coords >= 0) & (integer_coords < vt.nx)).all(1)
                ref_indices = torch.arange(integer_coords.shape[0], device='cuda')[kept]
                ref_coords = torch.cat([integer_coords[kept], torch.zeros((int(kept.sum()), 1), device='cuda', dtype=torch.long)], dim=1)
                with np.load(args.raw_out / token / 'repeat_00.npz') as captured:
                    matches_captured = np.array_equal(ref_coords.cpu().numpy(), captured['camera.pool.coords'])
                if not matches_captured:
                    raise RuntimeError('Reconstructed reference geometry differs from actual captured integer coordinates')
                ref_camera = ref_indices // (channels_depth * height * width)
                ref_spatial = ref_indices % (height * width)
                ref_products = features.reshape(cameras * height * width, -1)[ref_camera * height * width + ref_spatial] * depths.reshape(-1)[ref_indices, None]
                refmap_pool = base.bev_pool(ref_products.contiguous(), ref_coords, 1, 1, 360, 360)[:, :, 0]
                ascending = torch.argsort(idx)
                trtmap_pool = base.bev_pool(products[ascending].contiguous(), torch.from_numpy(coords).cuda()[ascending], 1, 1, 360, 360)[:, :, 0]
                geometry_arrays = {'reference_mapping': refmap_pool.cpu().numpy(), 'trt_mapping': trtmap_pool.cpu().numpy()}
                geometry_folder = raw_out / token
                geometry_folder.mkdir(parents=True, exist_ok=True)
                geometry_path = geometry_folder / (case + '.geometry.npz')
                np.savez(geometry_path, **geometry_arrays)
                geometry_result = {'scope': 'same half-representable features/depth; identical native FP32 pooling '
                    'kernel; both mappings ordered by ascending original ray source index; mapping-only comparison',
                    'reference_geometry_matches_captured_coordinates': matches_captured,
                    'reference_contributions': len(ref_indices), 'trt_contributions': len(indices),
                    'output_sha256': sha(geometry_path),
                    'mapping_only_difference': tensor_comparison(geometry_arrays['reference_mapping'], geometry_arrays['trt_mapping']),
                    'trt_map_summation_order_control': tensor_comparison(trtmap_pool.cpu().numpy(), pooled[:, :, 0].cpu().numpy())}

            elif module == 'fuser':
                x = model.fuser([tensors['camera'], tensors['lidar']])
                results = {'middle': first(model.decoder['neck'](model.decoder['backbone'](x)))}
            elif module == 'head':
                original = torch.Tensor.argsort
                captures = []
                def sort(tensor, *a, **kw):
                    result = original(tensor, *a, **kw)
                    if tensor.ndim == 2:
                        captures.append(result[:, :head.num_proposals].detach().cpu().numpy())
                    return result
                torch.Tensor.argsort = sort
                try:
                    pred = head(tensors['middle'], None)[0][0]
                finally:
                    torch.Tensor.argsort = original
                if len(captures) != 1:
                    raise RuntimeError('Expected one actual TopK argsort during head replay')
                source_ids = captures[0]
                classes = torch.nn.functional.one_hot(head.query_labels, num_classes=head.num_classes).permute(0, 2, 1)
                results = {name: pred[key][..., -head.num_proposals:]
                           for name, key in [('reg', 'center'), ('height', 'height'), ('dim', 'dim'),
                                             ('rot', 'rot'), ('vel', 'vel')]}
                results['score'] = pred['heatmap'][..., -head.num_proposals:].sigmoid() * pred['query_heatmap_score'] * classes
            else:
                raise ValueError('Unknown replay module ' + module)
            results = {k: v.detach().cpu().numpy() for k, v in results.items()}
            folder = raw_out / token
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / (case + '.outputs.npz')
            np.savez(target, **results)
            trt_path = path.with_name(case + '.outputs.npz')
            with np.load(trt_path) as trt:
                comparison = {k: tensor_comparison(v, trt[k]) for k, v in results.items()}
            row = {'token': token, 'case': case, 'input_sha256': sha(path),
                   'reference_output_sha256': sha(target), 'trt_output_sha256': sha(trt_path),
                   'nodes': comparison}
            if geometry_result is not None:
                row['geometry_mapping_replay'] = geometry_result
            if source_ids is not None:
                row['reference_query_source_indices'] = source_ids.tolist()
                query_path = path.with_name(case + '.query.npz')
                marked_path = path.with_name(case + '.marked.npz')
                if query_path.exists() and marked_path.exists():
                    with np.load(query_path) as query, np.load(marked_path) as marked, np.load(trt_path) as production_outputs:
                        marked_matches = all(k in marked and np.array_equal(marked[k], production_outputs[k]) for k in results)
                        row['marked_trt_outputs_equal_production'] = marked_matches
                        row['marked_trt_query_source_indices'] = query['source_indices'].tolist()
                        row['source_aligned_to_marked_trt'] = {
                            k: source_comparison(source_ids.reshape(-1), query['source_indices'].reshape(-1),
                                                 v[0].T, marked[k][0].T)
                            for k, v in results.items() if k in marked}
                        row['source_alignment_scope'] = ('production TRT verified by bitwise equality of all six marked outputs'
                            if marked_matches else 'marked TRT diagnostic graph only; production provenance unverified')

                row['alignment_warning'] = ('rowwise head differences are not object-aligned when query source indices differ; '
                                           'use diagnostic marked TRT TopK indices for source alignment')
            summary['cases'].append(row)
            write_json(out / 'summary.json', summary)
            print(token, case, 'reference replay completed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/deployment_4090.yaml')
    parser.add_argument('--raw-out', type=Path, default=ROOT / 'runs/deployment_4090_20261012_followup/reference')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/deployment_4090_20261012_followup/reference')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--replay-dir', type=Path, help='TRT replay tree containing token/*.inputs.npz; run local reference modules only')
    args = parser.parse_args()
    if args.repeat < 5 or args.warmup != 3:
        raise ValueError('This diagnosis requires >=5 repeats and production warmup=3')
    deploy = json.loads(args.config.read_text())
    packets = []
    for split in ['scene0061', 'mini_val']:
        packets.extend(json.loads((ROOT / deploy['splits'][split]).read_text()))
    case_path = ROOT / 'reports/deployment_4090_20261008/bevfusion_tensorrt/geometry_failure_cases.json'
    selected = select_packets(packets, json.loads(case_path.read_text()))
    if args.select_only:
        print(json.dumps([p['sample_token'] for p in selected], indent=2))
        return
    from bevfusion_4090 import CAMERAS, UPSTREAM_COMMIT, make_pipeline, packet_input, standardize, write_json
    source = ROOT / deploy['bevfusion']['source']
    sys.path.insert(0, str(source))
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    from mmdet3d.models import build_model
    from mmdet3d.core.bbox import LiDARInstance3DBoxes, Box3DMode
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('Unexpected upstream commit ' + commit)
    model_config = ROOT / deploy['bevfusion']['config']
    checkpoint_path = ROOT / deploy['bevfusion']['checkpoint']
    configs.load(str(model_config), recursive=True)
    cfg = Config(recursive_eval(configs), filename=str(model_config))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.encoders.camera.backbone.init_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    weights = checkpoint.get('state_dict', checkpoint)
    if all(k.startswith('module.') for k in weights):
        weights = {k[7:]: v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    del weights, checkpoint
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    pipeline, steps = make_pipeline(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    args.raw_out.mkdir(parents=True, exist_ok=True)
    head = model.heads['object']
    if head.test_cfg['nms_type'] is not None:
        raise ValueError('Output provenance supports the production no-post-NMS configuration only')
    vt = model.encoders['camera']['vtransform']
    if args.replay_dir is not None:
        reference_replay(args, model, torch, write_json)
        return
    # Production baseline executes before installing any instrumentation or patches.
    # Save only completed model outputs; there are no mid-forward CPU synchronizations.
    batches = {}
    production = {'scope': 'five uninstrumented forwards per fixed preprocessed input; '
                  'post-forward D2H only; no candidate-source capture or performance claim',
                  'tokens': []}
    def flatten_inputs(value, prefix='input', output=None):
        output = {} if output is None else output
        if torch.is_tensor(value):
            output[prefix] = value.detach().cpu().clone().numpy()
        elif isinstance(value, dict):
            for key, item in value.items():
                flatten_inputs(item, prefix + '.' + str(key), output)
        elif isinstance(value, (tuple, list)):
            for key, item in enumerate(value):
                flatten_inputs(item, prefix + '.' + str(key), output)
        return output
    with torch.inference_mode():
        for packet in selected:
            token = packet['sample_token']
            data = packet_input(packet)
            data.update(box_type_3d=LiDARInstance3DBoxes, box_mode_3d=Box3DMode.LIDAR)
            batch = scatter(collate([pipeline(data)], samples_per_gpu=1), [0])[0]
            batches[token] = batch
            frozen_inputs = flatten_inputs(batch)
            folder = args.raw_out / token
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(folder / 'production_inputs.npz', **frozen_inputs)
            entry = {'sample_token': token, 'input_identity': {k: array_stats(v) for k, v in frozen_inputs.items()},
                     'repeats': [], 'ordered_output_comparisons_to_repeat0': [],
                     'candidate_identity': 'not captured in uninstrumented production forwards; '
                                           'ordered tensor equality is not an object correspondence metric'}
            for _ in range(args.warmup):
                model(**batch)
            baseline = None
            for repeat in range(args.repeat):
                output = model(**batch)[0]
                torch.cuda.synchronize()
                output = {k: v.to('cpu') if hasattr(v, 'to') else v for k, v in output.items()}
                arrays = {'boxes': output['boxes_3d'].tensor.numpy(),
                          'scores': output['scores_3d'].numpy(), 'labels': output['labels_3d'].numpy()}
                boxes, audit = standardize(output, packet, cfg.object_classes)
                raw = folder / ('production_%02d.npz' % repeat)
                np.savez(raw, **arrays)
                entry['repeats'].append({'repeat': repeat, 'raw_path': str(raw), 'raw_sha256': sha(raw),
                    'native_boxes': arrays['boxes'].tolist(), 'scores': arrays['scores'].tolist(),
                    'labels': arrays['labels'].tolist(), 'standardized_boxes': boxes,
                    'standardization_audit': audit, 'tensor_stats': {k: array_stats(v) for k, v in arrays.items()}})
                if baseline is None:
                    baseline = arrays
                else:
                    entry['ordered_output_comparisons_to_repeat0'].append(
                        {'repeat': repeat, 'nodes': {k: tensor_comparison(baseline[k], v) for k, v in arrays.items()}})
                print(token, repeat, 'production baseline completed', flush=True)
            after_inputs = flatten_inputs(batch)
            if any(value.tobytes() != after_inputs[k].tobytes() for k, value in frozen_inputs.items()):
                raise RuntimeError('Production forward mutated fixed preprocessed inputs')
            write_json(args.out / (token + '.production.json'), entry)
            production['tokens'].append({'sample_token': token, 'report': token + '.production.json',
                'ordered_outputs_bitwise_equal': all(n.get('bitwise_equal', False)
                    for c in entry['ordered_output_comparisons_to_repeat0'] for n in c['nodes'].values())})
            write_json(args.out / 'production_summary.json', production)
    state = {'active': False, 'head_active': False, 'tensors': {}}

    def save(name, value):
        if not state['active']:
            return
        if torch.is_tensor(value):
            state['tensors'][name] = value.detach().cpu().clone().numpy()
        elif isinstance(value, dict):
            for k, v in value.items():
                save(name + '.' + str(k), v)
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                save(name + '.' + str(i), v)

    handles = []
    def hook(name, module):
        handles.append(module.register_forward_hook(lambda _m, _i, output: save(name, output)))
    hook('camera.neck', model.encoders['camera']['neck'])
    hook('camera.view_transform', vt)
    hook('lidar.bev', model.encoders['lidar']['backbone'])
    hook('fuser.bev', model.fuser)
    handles.append(head.register_forward_pre_hook(lambda _m, inputs: save('head.input', inputs[0][0] if isinstance(inputs[0], (tuple, list)) else inputs[0])))
    hook('head.raw', head)
    hook('head.dense_heatmap_logits', head.heatmap_head)
    def depth_hook(_m, _i, output):
        if state['active']:
            save('camera.feature', output[:, vt.D:vt.D + vt.C].permute(0, 2, 3, 1))
            save('camera.depth', output[:, :vt.D].softmax(1))
    handles.append(vt.depthnet.register_forward_hook(depth_hook))
    handles.append(vt.dtransform.register_forward_pre_hook(lambda _m, inputs: save('camera.sparse_depth', inputs[0])))
    handles.append(head.register_forward_pre_hook(lambda _m, _i: state.update(head_active=True)))
    handles.append(head.register_forward_hook(lambda _m, _i, _o: state.update(head_active=False)))
    base = importlib.import_module('mmdet3d.models.vtransforms.base')
    original_pool, original_argsort, original_decode = base.bev_pool, torch.Tensor.argsort, head.bbox_coder.decode
    original_setitem = torch.Tensor.__setitem__
    def pooling(features, coords, B, D, H, W):
        output = original_pool(features, coords, B, D, H, W)
        save('camera.pool.coords', coords)
        # Aggregate features are large; store hashes and contributions, not per-ray feature copies.
        save('camera.bev_pool', output)
        return output
    def argsort(tensor, *a, **kw):
        indices = original_argsort(tensor, *a, **kw)
        if state['active'] and state['head_active'] and tensor.ndim == 2:
            if 'query.source_indices' in state['tensors']:
                raise RuntimeError('More than one head argsort observed; provenance needs review')
            k = head.num_proposals
            save('query.pre_topk_scores', tensor)
            save('query.source_indices', indices[:, :k])
            save('query.selected_scores', tensor.gather(-1, indices[:, :k]))
            save('query.sort_margins', tensor.gather(-1, indices[:, :k]) - tensor.gather(-1, indices[:, 1:k + 1]))
        return indices
    def decode(heatmap, rot, dim, center, height, vel, filter=False):
        for name, value in zip(['score', 'rot', 'dim', 'reg', 'height', 'vel'],
                               [heatmap, rot, dim, center, height, vel]):
            save('head.' + name, value)
        result = original_decode(heatmap, rot, dim, center, height, vel, filter=filter)
        if state['active']:
            scores = heatmap.max(1).values
            bottom = height - dim[:, 2:3] * .5
            native_center = torch.cat([center, bottom], dim=1).permute(0, 2, 1)
            bounds = head.bbox_coder.post_center_range
            mask = (native_center >= bounds[:3]).all(2) & (native_center <= bounds[3:]).all(2)
            if head.bbox_coder.score_threshold:
                mask &= scores > head.bbox_coder.score_threshold
            sources = state['tensors']['query.source_indices'][0]
            output_sources = sources[mask[0].detach().cpu().numpy()]
            if len(output_sources) != len(result[0]['bboxes']):
                raise RuntimeError('Decoder candidate provenance mask does not match output')
            state['tensors']['output.source_indices'] = output_sources
        return result
    def setitem(tensor, index, value):
        # Observe the actual raster scatter arguments, without recomputing projection.
        if (state['active'] and tensor.ndim == 5 and tuple(tensor.shape[1:3]) == (6, 1)
                and isinstance(index, tuple) and len(index) == 5
                and isinstance(index[1], int) and torch.is_tensor(index[3])):
            prefix = 'camera.raster.' + str(index[1])
            save(prefix + '.rows', index[3])
            save(prefix + '.cols', index[4])
            save(prefix + '.depth_values', value)
        return original_setitem(tensor, index, value)
    base.bev_pool, torch.Tensor.argsort, head.bbox_coder.decode = pooling, argsort, decode
    torch.Tensor.__setitem__ = setitem
    config_files = [model_config] + [p / 'default.yaml' for p in model_config.parents if (p / 'default.yaml').exists()]
    identity = {'checkpoint_sha256': sha(checkpoint_path), 'upstream_commit': commit,
        'runner_sha256': sha(__file__), 'config_sha256': sha(args.config),
        'config_hierarchy': {str(p): sha(p) for p in config_files},
        'camera_order': CAMERAS, 'sweeps': 9, 'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda, 'device': torch.cuda.get_device_name(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'source_files': {str(p.relative_to(source)): sha(p) for p in [
            source / 'mmdet3d/models/heads/bbox/transfusion.py',
            source / 'mmdet3d/core/bbox/coders/transfusion_bbox_coder.py',
            source / 'mmdet3d/models/vtransforms/base.py',
            source / 'mmdet3d/models/vtransforms/depth_lss.py']},
        'production_config_unchanged': True, 'warmup_per_token': args.warmup,
        'repeat_per_token': args.repeat, 'performance_measurement': False,
        'instrumentation': 'CPU copies/hooks synchronize execution; no wall-clock speed claim'}
    write_json(args.out / 'identity.json', identity)
    write_json(args.out / 'resolved_config.json', {'model': cfg.model, 'pipeline': steps})
    summary = {'identity': identity, 'tokens': [], 'node_layouts': {
        'camera.feature': 'NHWC; six cameras folded into N', 'camera.depth': 'NDHW',
        'camera.sparse_depth': 'N1HW', 'camera.bev_pool': 'BCZXY; collapse Z into C to compare TRT NCHW',
        'camera.view_transform': 'NCHW', 'lidar.bev': 'NCHW', 'fuser.bev': 'NCHW',
        'head.input': 'NCHW; TRT fuser middle/head input',
        'query.pre_topk_scores': 'B,(class*HW+spatial index); after local maximum suppression',
        'query.source_indices': 'actual argsort class*HW+spatial index; no row identity inference',
        'head.*': 'BCQ; score includes seed class mask and query heatmap score',
        'output.native_boxes': 'Nx9 lidar bottom-center xyz, dimensions, yaw, vx, vy'},
        'gaps': ['No TRT raw-fuser binding without debug export',
                 'Geometry per-interval statistics derived from reference integer pooling coordinates; TRT ranks differ by representation'],
        'comparison_scope': 'each repeat versus repeat0; all finite elements use fixed tensor tolerances'}
    try:
        with torch.inference_mode():
            for packet in selected:
                token = packet['sample_token']
                folder = args.raw_out / token
                folder.mkdir(parents=True, exist_ok=True)
                batch = batches[token]
                state.update(active=True, tensors={})
                save('input', batch)
                input_tensors = state['tensors']
                np.savez(folder / 'preprocessed_inputs.npz', **input_tensors)
                input_identity = {k: array_stats(v) for k, v in input_tensors.items()}
                state.update(active=False, tensors={})
                for _ in range(args.warmup):
                    model(**batch)
                torch.cuda.synchronize()
                token_report = {'sample_token': token, 'input_identity': input_identity,
                    'input_file_sha256': sha(folder / 'preprocessed_inputs.npz'),
                    'sensor_files': {p: sha(p) for p in sorted({s['path'] for s in list(packet['sensors'].values()) + packet['lidar_sweeps'][:9]})},
                    'actual_sweeps': [s['path'] for s in packet['lidar_sweeps'][:9]],
                    'repeat': [], 'comparisons_to_repeat0': []}
                baseline = None
                for repeat in range(args.repeat):
                    state.update(active=True, tensors={}, head_active=False)
                    result = model(**batch)[0]
                    torch.cuda.synchronize()
                    result = {k: v.to('cpu') if hasattr(v, 'to') else v for k, v in result.items()}
                    save('output.native_boxes', result['boxes_3d'].tensor)
                    save('output.scores', result['scores_3d'])
                    save('output.labels', result['labels_3d'])
                    save('input_after', batch)
                    tensors = state['tensors']
                    state['active'] = False
                    for key, value in input_tensors.items():
                        after = tensors.pop(key.replace('input.', 'input_after.', 1))
                        if value.tobytes() != after.tobytes():
                            raise RuntimeError('Fixed preprocessed input mutated: ' + key)
                    required = ['camera.feature', 'camera.depth', 'camera.bev_pool', 'camera.view_transform',
                                'lidar.bev', 'fuser.bev', 'query.source_indices', 'output.source_indices']
                    missing = [k for k in required if k not in tensors]
                    if missing:
                        raise RuntimeError('Missing required instrumentation: ' + str(missing))
                    boxes, audit = standardize(result, packet, cfg.object_classes)
                    output_sources = tensors['output.source_indices']
                    for i, box in enumerate(boxes):
                        box['query_source_index'] = int(output_sources[i])
                    native = [{'query_source_index': int(s), 'tensor': b.tolist(), 'score': float(score), 'label': int(label)}
                              for s, b, score, label in zip(output_sources, tensors['output.native_boxes'],
                                                          tensors['output.scores'], tensors['output.labels'])]
                    out_file = folder / ('repeat_%02d.npz' % repeat)
                    np.savez(out_file, **tensors)
                    coordinates = tensors['camera.pool.coords']
                    _, counts = np.unique(coordinates, axis=0, return_counts=True)
                    raster_collisions = {}
                    for camera in range(6):
                        prefix = 'camera.raster.' + str(camera)
                        pixels = tensors[prefix + '.rows'] * vt.image_size[1] + tensors[prefix + '.cols']
                        order = np.argsort(pixels, kind='stable')
                        _, starts, count = np.unique(pixels[order], return_index=True, return_counts=True)
                        depths = tensors[prefix + '.depth_values'][order]
                        depth_range = np.maximum.reduceat(depths, starts) - np.minimum.reduceat(depths, starts)
                        raster_collisions[CAMERAS[camera]] = {
                            'projected_points': int(len(pixels)), 'occupied_pixels': int(len(count)),
                            'collision_pixels': int((count > 1).sum()),
                            'collision_pixels_with_distinct_depth': int(((count > 1) & (depth_range > 0)).sum()),
                            'collision_depth_range': quantiles(depth_range[count > 1])}
                    entry = {'repeat': repeat, 'raw_path': str(out_file), 'raw_sha256': sha(out_file),
                        'tensor_stats': {k: array_stats(v, k) for k, v in tensors.items()},
                        'sparse_depth_projection_collisions': raster_collisions,
                        'pool_geometry': {'valid_contributions': int(len(coordinates)), 'occupied_voxels': int(len(counts)),
                                          'contributions_per_voxel': quantiles(counts)},
                        'native_boxes': native, 'standardized_boxes': boxes, 'standardization_audit': audit,
                        'query_seed': {'source_indices': tensors['query.source_indices'][0].tolist(),
                                       'scores': tensors['query.selected_scores'][0].tolist(),
                                       'sort_margins': tensors['query.sort_margins'][0].tolist()}}
                    token_report['repeat'].append(entry)
                    if baseline is None:
                        baseline = tensors
                    else:
                        comparisons = {k: tensor_comparison(baseline[k], v) for k, v in tensors.items()}
                        aligned = source_comparison(baseline['output.source_indices'], output_sources,
                                                    baseline['output.native_boxes'], tensors['output.native_boxes'])
                        token_report['comparisons_to_repeat0'].append({'repeat': repeat, 'nodes': comparisons,
                                                                       'native_boxes_by_source': aligned})
                    write_json(args.out / (token + '.json'), token_report)
                    print(token, repeat, 'captured', len(tensors), 'nodes', flush=True)
                summary['tokens'].append({'sample_token': token, 'report': token + '.json',
                    'all_nodes_bitwise_equal': all(n.get('bitwise_equal', False) for c in token_report['comparisons_to_repeat0'] for n in c['nodes'].values()),
                    'all_nodes_within_tensor_tolerance': all(n['passed'] for c in token_report['comparisons_to_repeat0'] for n in c['nodes'].values()),
                    'source_aligned_native_max_abs': max(c['native_boxes_by_source']['aligned_max_abs'] or 0 for c in token_report['comparisons_to_repeat0'])})
                write_json(args.out / 'summary.json', summary)
    finally:
        base.bev_pool, torch.Tensor.argsort, head.bbox_coder.decode = original_pool, original_argsort, original_decode
        torch.Tensor.__setitem__ = original_setitem
        for handle in handles:
            handle.remove()


if __name__ == '__main__':
    main()
