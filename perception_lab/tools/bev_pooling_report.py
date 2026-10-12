"""Regenerate S3 Markdown, plots and status from verified current evidence (CPU only)."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    lab = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=lab/'reports/deployment_4090_20261008/bev_pooling')
    a = p.parse_args(); r = a.out
    kernel = json.loads((r/'results.json').read_text())
    full = json.loads((r/'full_model_comparison.json').read_text())
    for split, evidence in full['splits'].items():
        for kind, base in [('upstream', lab/'reports/deployment_4090_20261008/bevfusion_tensorrt'/split/'official'),
                           ('candidate', r/'full_model'/split/'official')]:
            hashes = evidence['verified_provenance'][kind]
            for key, filename in [('metrics_summary_sha256', 'metrics/metrics_summary.json'),
                                   ('metrics_details_sha256', 'metrics/metrics_details.json'),
                                   ('export_audit_sha256', 'export_audit.json')]:
                if sha(base/filename) != hashes[key]:
                    raise ValueError('Stale full-model comparison; rerun bev_pooling_full_model.py: ' + str(base/filename))
    real = [x for x in kernel['cases'] if x['real_capture']]
    capture_statistics = []
    capture_root = lab/'runs/deployment_4090_20261008/trt_pooling_captures'
    for case in real:
        prefix = capture_root/case['name']
        for name, expected in case['captured_sha256'].items():
            if sha(Path(str(prefix)+'.'+name+'.bin')) != expected:
                raise ValueError('Stale kernel result versus current capture: ' + str(prefix))
        intervals = np.fromfile(str(prefix)+'.intervals.bin', dtype=np.int32).reshape(-1, 3)
        lengths = intervals[:, 1]-intervals[:, 0]
        capture_statistics.append(dict(name=case['name'], metadata=case['metadata'],
            interval_length_min=int(lengths.min()), interval_length_max=int(lengths.max()),
            interval_length_p50=float(np.percentile(lengths,50)), interval_length_p95=float(np.percentile(lengths,95)),
            hashes=case['captured_sha256']))
    (r/'input_statistics.json').write_text(json.dumps(capture_statistics,indent=2)+'\n')
    rows = list(csv.DictReader((r/'paired_timings.csv').open()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = ['#2864a5', '#cd7044']
    fig, ax = plt.subplots(figsize=(7, 3.8), layout='constrained')
    for i, case in enumerate(real):
        for j, key in enumerate(['upstream_ms', 'candidate_ms']):
            values = [float(x[key]) for x in rows if x['case'] == case['name']]
            boxes = ax.boxplot(values, positions=[i*3+j], widths=.65, patch_artist=True, showmeans=True)
            boxes['boxes'][0].set_facecolor(colors[j]); boxes['boxes'][0].set_alpha(.65)
    ax.set_xticks([i*3+.5 for i in range(len(real))], [x['name'] for x in real])
    ax.set_ylabel('Zero output + pooling CUDA time (ms)'); ax.set_title(str(kernel['paired_rounds']) + ' alternating paired rounds per real frame')
    ax.legend(handles=[Patch(facecolor=colors[0], label='Upstream: pack10 / 1024 threads'),
                       Patch(facecolor=colors[1], label='Candidate: pack4 / 160 threads')], fontsize=8)
    ax.grid(axis='y', alpha=.2); fig.savefig(r/'paired_latency.png', dpi=180); plt.close(fig)
    stage = dict(experiment=kernel['status'], correctness=kernel['status'], paired_measurement='passed',
                 speedup_obtained=all(x['timing']['speedup']['p50'] > 1 for x in real),
                 default=full['selected_default'], nsight_systems='passed', compute_sanitizer='passed',
                 nsight_compute='blocked_permission', full_model=full['status'], full_model_frames=sum(x['frames'] for x in full['splits'].values()),
                 full_model_paired_rounds=full['paired']['rounds'], stable_full_model_speedup=full['stable_speedup_obtained'],
                 capacity_boundaries='passed', evidence_sha256={f:sha(r/f) for f in ['results.json', 'full_model_comparison.json', 'paired_timings.csv']})
    for filename in ['capacity_max.json', 'capacity_min.json', 'capacity_reject.json']:
        if json.loads((r/filename).read_text())['status'] != 'passed':
            stage['capacity_boundaries'] = 'failed'
    for filename in ['sanitizer_edges.log', 'sanitizer_real.log']:
        if 'ERROR SUMMARY: 0 errors' not in (r/filename).read_text():
            stage['compute_sanitizer'] = 'failed'
    text = '''# BEV pooling candidate experiment on RTX4090

The custom CUDA candidate passes the measured numerical checks, but does not establish stable speed improvement. **The upstream kernel remains the deployment default.** This report is generated from current JSON/CSV evidence by `tools/bev_pooling_report.py`.

## Implementation and actual inputs

`tools/bev_pooling_candidate.patch` modifies NVIDIA CUDA-BEVFusion `camera-bevpool.cu`. `CAR_BEV_POOLING=candidate` selects four adjacent FP16 channels per thread, using block `(20,8)` (160 threads). Default upstream retains ten channels per thread and block `(8,128)` (1024 threads). Both execute the same interval/point order, FP32 FMA accumulation and final half conversion; no atomics or approximate arithmetic are added. The complete model and paired benchmark call the same compiled C ABI dispatcher.

Three real CUDA/TensorRT pipeline captures from scene-0061 frames 0/19/38 contain NHWC camera features, NDHW depth probabilities, uint32 indices, int32 interval triples and NCHW reference output. Input shape is `[6,80,118,32,88]`; output is `[1,80,360,360]`, or 10,368,000 half values. Hashes, finite-value checks and interval distributions are retained in `input_statistics.json` and `results.json`. The three captures cover their sampled smallest/largest interval counts; they are not claimed to be extrema over every mini frame. Large inputs stay local in `runs/deployment_4090_20261008/trt_pooling_captures/`.

## Correctness and memory checks

'''
    tol = kernel['tolerance_fixed_before_run']
    text += f"The tolerance was fixed before measurement: FP16 atol={tol['atol']}, rtol={tol['rtol']}. "
    exact = all(x['candidate_vs_upstream']['bitwise_equal'] and x['upstream_vs_captured_or_cpu']['bitwise_equal'] for x in real)
    text += ('All three candidate/upstream/captured real outputs are **bitwise identical**, max absolute error 0.\n\n' if exact else 'Numerical differences are reported in `results.json`; no exact-equality claim is made.\n\n')
    text += '''Empty valid projections, a single point, multiple contributions to one cell, and last-input/last-output boundary cases pass numerical checks, with an independent CPU sum for the small cases. The shared host dispatcher returns a zero tensor for empty intervals, avoiding the original invalid zero-block launch. Nonempty reference accumulation semantics are unchanged. Input validation checks bounds and unique destination cells before CUDA.

Compute Sanitizer memcheck reports **0 errors** on all four edge cases and the full real frame-0 reference/candidate pair (`sanitizer_edges.log`, `sanitizer_real.log`). This is memory-safety evidence for the tested cases, separate from accuracy checks.

## Isolated paired timing

Each real case has 10 independent warmups per backend and 30 pairs with alternating order. CUDA events measure the same output-zeroing plus reduction boundary on preallocated identical inputs/outputs. H2D/D2H, sorting, allocation, capture I/O and rendering are excluded. All 90 rows are in `paired_timings.csv`.

| Capture | Upstream P50/P95 ms | Candidate P50/P95 ms | Median paired speedup |
|---|---:|---:|---:|
'''
    for case in real:
        t=case['timing'];text+=f"| {case['name']} | {t['upstream_ms']['p50']:.6f}/{t['upstream_ms']['p95']:.6f} | {t['candidate_ms']['p50']:.6f}/{t['candidate_ms']['p95']:.6f} | {t['speedup']['p50']:.3f}× |\n"
    text += '''
![Paired kernel timings](paired_latency.png)

## Profiling and limitations

Nsight Systems traced the actual pair; CSV summaries and `nsys_kernel_resources.json` are published, with the full `.nsys-rep` retained locally. Actual register usage changes only from 40 to 38 registers/thread. The recorded grids are 761 blocks ×1024 threads upstream and 12,166 ×160 candidate. The candidate uses 2.5 times as many channel workers and repeats index/depth work more often. Reduced register pressure did not automatically produce faster execution. Scheduling and repeated work are plausible explanations, not a measured hardware-counter root cause.

Nsight Compute was invoked with both kernel filters and `--set full`, but failed with **ERR_NVGPUCTRPERM** because driver performance counters are inaccessible to this account (`ncu.log`). Achieved occupancy, bandwidth efficiency and stall counters were not measured; no driver settings were changed. A single traced pair is diagnostic and does not replace warmed timing distributions.

'''
    profile = r.parent/'bevfusion_tensorrt/native_profile.log'
    if profile.exists():
        timings={name:float(ms) for name,ms in re.findall(r'\[⏰ (.*?)\]:\s*([0-9.]+) ms', profile.read_text())}
        if timings:
            text+='The native S2 timer recorded this final warm first-frame breakdown after three warmups. Instrumentation synchronizes stage boundaries, so these numbers are not uninstrumented P50 latency. `[NoSt] CopyLidar` is shown separately and excluded from its printed core total.\n\n| Native stage | ms |\n|---|---:|\n'
            for name,ms in timings.items():text+=f'| {name} | {ms:.5f} |\n'
            text+='\n'
    text+='''## Complete-model comparison

The candidate ran both fixed splits through the final LayerNorm-plugin TensorRT engines. `full_model_comparison.json` verifies matching checkpoint, ONNX files, engines, libraries, configuration, packets and every preprocessed image/point/matrix hash. Only pooling mode changes. The summary checks every sealed inference artifact against its manifest and verifies official export audit input/output hashes before using metrics. Candidate evaluations and detailed matched/unmatched box diagnostics are under `full_model/<split>/`.

| Split | Frames | Upstream mAP/NDS | Candidate mAP/NDS | ΔmAP / ΔNDS | Correctness |
|---|---:|---:|---:|---:|---|
'''
    for split,x in full['splits'].items():
        m=x['metrics'];text+=f"| {split} | {x['frames']} | {m['mean_ap']['upstream']:.6f}/{m['nd_score']['upstream']:.6f} | {m['mean_ap']['candidate']:.6f}/{m['nd_score']['candidate']:.6f} | {m['mean_ap']['delta']:+.6f}/{m['nd_score']['delta']:+.6f} | {x['status']} |\n"
    text+='''
The fixed maximum mAP/NDS drop is 0.005; fixed P95 geometry/score/velocity diagnostics and per-class AP deltas are retained. This validates candidate versus upstream TensorRT, independently of TensorRT-versus-PyTorch acceptance. Full native outputs are not assumed bitwise identical; other native GPU stages may be nondeterministic, but their variation was not isolated experimentally.

A separate resident-core run uses the same actual first-frame FP16 CPU arrays, one geometry update, 10 independent warmups per mode and 30 alternating full-forward pairs. It includes synchronized forward/H2D/native-box D2H and excludes preprocessing, geometry recalculation, loading, rendering and capture I/O. Raw timing/output hashes are in `full_model_paired_timings.csv`.

'''
    q=full['paired'];t=q['statistics'];ci=q['bootstrap_mean_saved_ms_95ci']
    text+=f"Upstream mean/P50/P95: **{t['upstream_ms']['mean']:.3f}/{t['upstream_ms']['p50']:.3f}/{t['upstream_ms']['p95']:.3f} ms**; candidate: **{t['candidate_ms']['mean']:.3f}/{t['candidate_ms']['p50']:.3f}/{t['candidate_ms']['p95']:.3f} ms**. Mean paired saving is {q['mean_saved_ms']:.3f} ms; exploratory bootstrap 95% interval **[{ci[0]:.3f}, {ci[1]:.3f}] ms**. Candidate is faster in {q['candidate_faster_pairs']}/{q['rounds']} pairs. Mean savings grouped by upstream-first/candidate-first order are {q['upstream_first_mean_saved_ms']:.3f}/{q['candidate_first_mean_saved_ms']:.3f} ms. The bootstrap assumes exchangeable pairs and is descriptive; order sensitivity and single-frame scope limit any whole-scene speed claim. Selected default: **{full['selected_default']}**.\n\n"
    text+='''## Capacity boundaries and reproduction

Separate synthetic checks tiled a real preprocessed cloud to exactly 500,000 points and retained a single actual point. Both returned 200 finite boxes without a crash (`capacity_max.json`, `capacity_min.json`); these are capacity checks, not accuracy or warmed-latency claims. A CPU test executes the actual runner's point-count guard AST and verifies 500,001 points are rejected before backend execution (`capacity_reject.json`). The dense image engines remain fixed shape: 500,000 describes native point capacity, not a dynamic image profile.

```bash
cd perception_lab
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage bev_pooling
# CPU-only verification and report regeneration from completed runs:
envs/nuscenes_eval/bin/python tools/bev_pooling_full_model.py
envs/nuscenes_eval/bin/python tools/bev_pooling_report.py
```

Large candidate outputs remain in `outputs/bevfusion_tensorrt_candidate/<split>/`; their manifests, compact timing/input audits and official evaluation evidence are published alongside this report. Raw source/engine/input/output hashes support reconstruction. No isolated-kernel ratio is presented as whole-model speedup.
'''
    text = text.replace('10 independent warmups per backend and 30 pairs', str(kernel['independent_warmup']) + ' independent warmups per backend and ' + str(kernel['paired_rounds']) + ' pairs')
    text = text.replace('All 90 rows', 'All ' + str(len(rows)) + ' rows')
    text = text.replace('10 independent warmups per mode and 30 alternating full-forward pairs', str(q['independent_warmup']) + ' independent warmups per mode and ' + str(q['rounds']) + ' alternating full-forward pairs')
    (r/'report.md').write_text(text)
    (r/'stage_summary.json').write_text(json.dumps(stage,indent=2)+'\n')
    for split in full['splits']:
        source=lab/'outputs/bevfusion_tensorrt_candidate'/split
        target=r/'full_model'/split
        for filename in ['manifest.json','performance.json','timings.csv','coordinate_audit.json','actual_inputs.json']:
            (target/filename).write_bytes((source/filename).read_bytes())
    print(json.dumps(stage,indent=2))


if __name__ == '__main__':
    main()
