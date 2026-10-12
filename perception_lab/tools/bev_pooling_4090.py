"""Paired benchmark of actual CUDA-BEVFusion FP16 pooling and CAR tile4.

Run with tools/bevfusion_trt_env.sh sourced. Both benchmark paths call the exact
kernels selected by the full model's CAR_BEV_POOLING environment switch.
"""
import argparse
import ctypes
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_capture(prefix):
    metadata = json.loads(Path(str(prefix) + '.json').read_text())
    arrays, hashes = {}, {}
    for name, dtype in [('features', np.float16), ('depth', np.float16), ('indices', np.uint32),
                        ('intervals', np.int32), ('output', np.float16)]:
        path = Path(str(prefix) + '.' + name + '.bin')
        arrays[name] = np.fromfile(path, dtype=dtype)
        hashes[name] = sha256(path)
    arrays['indices'] = arrays['indices'].astype(np.int64).astype(np.int32)
    arrays['intervals'] = arrays['intervals'].reshape(-1, 3)
    N, C, D, H, W = metadata['shape']
    assert arrays['features'].size == N * C * H * W
    assert arrays['depth'].size == N * D * H * W
    assert arrays['indices'].size == N * D * H * W
    assert len(arrays['intervals']) == metadata['num_intervals']
    assert arrays['output'].size == C * metadata['out_h'] * metadata['out_w']
    return dict(name=prefix.name, metadata=metadata, arrays=arrays, sha256=hashes, real_capture=True)


def validate_case(case):
    m, x = case['metadata'], case['arrays']
    N, C, D, H, W = m['shape']
    assert C == 80 and min(N, D, H, W, m['out_h'], m['out_w']) > 0
    intervals = x['intervals']
    assert intervals.shape == (m['num_intervals'], 3)
    assert np.isfinite(x['features']).all() and np.isfinite(x['depth']).all()
    if len(intervals):
        assert (intervals[:, 0] >= 0).all()
        assert (intervals[:, 1] >= intervals[:, 0]).all()
        assert (intervals[:, 1] <= len(x['indices'])).all()
        assert (intervals[:, 2] >= 0).all() and (intervals[:, 2] < m['out_h'] * m['out_w']).all()
        assert len(set(intervals[:, 2].tolist())) == len(intervals), 'Intervals must have unique output cells'
        for begin, end, _ in intervals:
            selected = x['indices'][begin:end]
            assert (selected >= 0).all() and (selected < N * D * H * W).all()


def edge_cases():
    rng = np.random.default_rng(4090)
    features = rng.normal(size=(6, 80)).astype(np.float16)
    depth = np.linspace(.1, .9, 12).astype(np.float16)
    specs = [('empty', [], []), ('single', [0], [[0, 1, 0]]),
             ('collision', [0, 6, 1, 7], [[0, 4, 2]]),
             ('boundary', [11, 0], [[0, 1, 5], [1, 2, 0]])]
    for name, indices, intervals in specs:
        array = np.asarray(intervals, dtype=np.int32).reshape(-1, 3)
        expected = np.zeros((80, 6), dtype=np.float16)
        for begin, end, cell in array:
            acc = np.zeros(80, dtype=np.float64)
            for i in range(begin, end):
                index = indices[i]
                acc += features[index % 6].astype(np.float64) * float(depth[index])
            expected[:, cell] = acc.astype(np.float16)
        yield dict(name=name, metadata=dict(shape=[1, 80, 2, 2, 3], out_h=2, out_w=3, num_intervals=len(array)),
                   arrays=dict(features=features.reshape(-1), depth=depth, indices=np.asarray(indices, dtype=np.int32),
                               intervals=array, output=expected.reshape(-1)), real_capture=False)


class Pool:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library.resolve()))
        self.run = self.library.car_bev_pool_forward
        self.run.restype = ctypes.c_int
        self.run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]

    def prepare(self, case):
        validate_case(case)
        self.metadata = case['metadata']
        self.arrays = {k: torch.from_numpy(v.copy()).cuda() for k, v in case['arrays'].items()}
        self.outputs = [torch.empty_like(self.arrays['output']) for _ in range(2)]
        torch.cuda.synchronize()

    def __call__(self, candidate):
        x, m = self.arrays, self.metadata
        _, C, D, H, W = m['shape']
        out = self.outputs[candidate]
        code = self.run(x['features'].data_ptr(), x['depth'].data_ptr(), x['indices'].data_ptr(),
                        x['intervals'].data_ptr(), m['num_intervals'], C, D, H, W, m['out_h'], m['out_w'],
                        out.data_ptr(), candidate, torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError('CUDA pooling error code %d' % code)
        return out


def compare(a, b, atol, rtol):
    x, y = a.float(), b.float()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    error = (x - y).abs()
    bad = error > (atol + rtol * y.abs())
    quantiles = torch.quantile(error, torch.tensor([.5, .95, .99], device=error.device)).cpu().tolist()
    return dict(finite=finite, elements=a.numel(), violations=int(bad.sum()),
                max_absolute=float(error.max()), p50_absolute=quantiles[0], p95_absolute=quantiles[1], p99_absolute=quantiles[2],
                max_relative=float((error / y.abs().clamp_min(1e-8)).max()),
                bitwise_equal=bool(torch.equal(a, b)), passed=finite and not bool(bad.any()))


def measure(pool, backend):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    begin.record()
    pool(backend)
    end.record()
    end.synchronize()
    return begin.elapsed_time(end)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, action='append', default=[])
    p.add_argument('--library', type=Path, default=Path('third_party/Lidar_AI_Solution/CUDA-BEVFusion/build/libbevfusion_core.so'))
    p.add_argument('--config', type=Path, default=Path('configs/deployment_4090.yaml'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--profile-only', action='store_true', help='One actual reference/candidate pair for Nsight')
    p.add_argument('--edges-only', action='store_true')
    a = p.parse_args()
    config = json.loads(a.config.read_text())['pooling']
    rounds, warmup = config['paired_rounds'], config['warmup']
    atol, rtol = config['fp16_atol'], config['fp16_rtol']
    assert rounds >= 30
    if not a.capture and not a.edges_only:
        raise ValueError('Real captures are required unless explicitly running edge checks only')
    a.out.mkdir(parents=True, exist_ok=True)
    pool = Pool(a.library)
    cases = list(edge_cases()) if not a.profile_only else []
    cases += [load_capture(prefix) for prefix in a.capture]
    rows, evidence = [], []
    for case in cases:
        pool.prepare(case)
        if a.profile_only:
            pool(0); pool(1); torch.cuda.synchronize()
            print('Profile pair:', case['name'])
            continue
        reference, candidate = pool(0), pool(1)
        torch.cuda.synchronize()
        check = compare(candidate, reference, atol, rtol)
        original_check = compare(reference, pool.arrays['output'], atol, rtol)
        record = dict(name=case['name'], real_capture=case['real_capture'], metadata=case['metadata'],
                      candidate_vs_upstream=check, upstream_vs_captured_or_cpu=original_check,
                      captured_sha256=case.get('sha256'), interval_length_min=int(np.min(case['arrays']['intervals'][:,1]-case['arrays']['intervals'][:,0])) if len(case['arrays']['intervals']) else 0,
                      interval_length_max=int(np.max(case['arrays']['intervals'][:,1]-case['arrays']['intervals'][:,0])) if len(case['arrays']['intervals']) else 0)
        evidence.append(record)
        if not check['passed'] or not original_check['passed']:
            (a.out / 'correctness_failed.json').write_text(json.dumps(evidence, indent=2))
            raise AssertionError('Pooling numerical failure: ' + case['name'])
        if not case['real_capture']:
            print(case['name'], 'correctness passed')
            continue
        for backend in [0, 1]:
            for _ in range(warmup):
                pool(backend)
            torch.cuda.synchronize()
        for round_id in range(rounds):
            order = [0, 1] if round_id % 2 == 0 else [1, 0]
            measurements = {backend: measure(pool, backend) for backend in order}
            rows.append(dict(case=case['name'], round=round_id, order='upstream-first' if order[0] == 0 else 'candidate-first',
                             upstream_ms=measurements[0], candidate_ms=measurements[1], speedup=measurements[0]/measurements[1]))
        local = [r for r in rows if r['case'] == case['name']]
        record['timing'] = {key: dict(mean=float(np.mean([r[key] for r in local])),
                                      p50=float(np.percentile([r[key] for r in local], 50)),
                                      p95=float(np.percentile([r[key] for r in local], 95)))
                            for key in ['upstream_ms', 'candidate_ms', 'speedup']}
        print(json.dumps(record), flush=True)
    if a.profile_only:
        return
    if rows:
        with (a.out / 'paired_timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(status='passed', experiment='CUDA-BEVFusion actual depth-weighted FP16 pooling',
                  tolerance_fixed_before_run=dict(atol=atol, rtol=rtol, config_sha256=sha256(a.config)),
                  paired_rounds=rounds, independent_warmup=warmup, cases=evidence,
                  boundary_empty='Both host dispatches return a zero tensor without an invalid zero-block launch; original upstream did not guard empty intervals.',
                  timing_boundary='same preallocated GPU inputs/outputs; zero output memset plus pooling kernel, CUDA event synchronized; no H2D/D2H, sort, allocation or capture I/O',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  library_sha256=sha256(a.library), adapter_sha256=sha256(__file__), default='upstream',
                  full_model_validation='pending separate paired full-model evaluation')
    (a.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
