"""Resident full CUDA-BEVFusion pooling-switch measurement and comparison summary."""
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def paired_forwards(core, images, points, matrices, rounds, warmup, out):
    """Same CPU FP16 inputs and resident engines; synchronized wall-time pairs."""
    import torch
    if rounds < 30:
        raise ValueError('At least 30 paired full-model rounds are required')
    images = np.ascontiguousarray(images, dtype=np.float16)
    points = np.ascontiguousarray(points, dtype=np.float16)
    out = Path(out)
    prior = os.environ.get('CAR_BEV_POOLING')
    capture = os.environ.pop('CAR_BEV_POOL_CAPTURE', None)
    rows = []
    core.set_timer(False)
    core.update(*matrices)
    try:
        for mode in ['upstream', 'candidate']:
            os.environ['CAR_BEV_POOLING'] = mode
            for _ in range(warmup):
                core.forward(images, points, False, False)
            torch.cuda.synchronize()
        for round_id in range(rounds):
            order = ['upstream', 'candidate'] if round_id % 2 == 0 else ['candidate', 'upstream']
            row = dict(round=round_id, order='-then-'.join(order))
            outputs = {}
            for mode in order:
                os.environ['CAR_BEV_POOLING'] = mode
                torch.cuda.synchronize()
                start = time.perf_counter()
                raw = np.asarray(core.forward(images, points, False, False)).copy()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if raw.ndim != 2 or raw.shape[1] != 11 or not np.isfinite(raw).all():
                    raise ValueError('Non-finite or malformed paired full-model output')
                row[mode + '_ms'] = elapsed
                row[mode + '_boxes'] = len(raw)
                row[mode + '_sha256'] = hashlib.sha256(raw.tobytes()).hexdigest()
                outputs[mode] = raw
            row['speedup'] = row['upstream_ms'] / row['candidate_ms']
            row['native_bitwise_equal'] = np.array_equal(outputs['upstream'], outputs['candidate'])
            rows.append(row)
        with (out / 'full_model_paired_timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        statistics = {key: dict(mean=float(np.mean([r[key] for r in rows])),
                                p50=float(np.percentile([r[key] for r in rows], 50)),
                                p95=float(np.percentile([r[key] for r in rows], 95)))
                      for key in ['upstream_ms', 'candidate_ms', 'speedup']}
        result = dict(status='measured', rounds=rounds, independent_warmup=warmup,
                      timing='synchronized resident-core forward, including H2D and native box D2H; same preprocessed CPU FP16 image/point arrays; geometry updated once; no capture I/O, model reload, preprocessing or rendering',
                      statistics=statistics, all_native_outputs_bitwise_equal=all(r['native_bitwise_equal'] for r in rows),
                      input_sha256={name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in [('images', images), ('points', points)]},
                      gpu=torch.cuda.get_device_name())
        (out / 'full_model_paired_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        if prior is None:
            os.environ.pop('CAR_BEV_POOLING', None)
        else:
            os.environ['CAR_BEV_POOLING'] = prior
        if capture is not None:
            os.environ['CAR_BEV_POOL_CAPTURE'] = capture
    return result


def verify_inference_and_evaluation(directory, manifest, evaluation):
    from bevfusion_4090 import digest
    sealed = manifest.get('outputs_sha256', {})
    if manifest.get('status') != 'passed' or 'predictions.json' not in sealed:
        raise ValueError('Missing successful sealed inference manifest: ' + str(directory))
    for filename, expected in sealed.items():
        path = directory/filename
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Inference artifact hash mismatch: ' + str(path))
    prediction_hash = digest(directory/'predictions.json')
    audit_path = evaluation/'export_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit.get('input_sha256') != prediction_hash:
        raise ValueError('Evaluation used different predictions: ' + str(audit_path))
    if audit.get('output_sha256') != digest(evaluation/'official_predictions.json'):
        raise ValueError('Official export hash mismatch: ' + str(evaluation))
    return dict(predictions_sha256=prediction_hash,
        inference_manifest_sha256=digest(directory/'manifest.json'),
        export_audit_sha256=digest(audit_path),
        official_predictions_sha256=digest(evaluation/'official_predictions.json'),
        metrics_summary_sha256=digest(evaluation/'metrics/metrics_summary.json'),
        metrics_details_sha256=digest(evaluation/'metrics/metrics_details.json'))


def main():
    import argparse
    from bevfusion_4090 import digest, write_json
    from bevfusion_compare import compare
    lab = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=lab/'configs/deployment_4090.yaml')
    p.add_argument('--reference-dir', type=Path, default=lab/'outputs/bevfusion_tensorrt')
    p.add_argument('--candidate-dir', type=Path, default=lab/'outputs/bevfusion_tensorrt_candidate')
    p.add_argument('--reference-report', type=Path, default=lab/'reports/deployment_4090_20261008/bevfusion_tensorrt')
    p.add_argument('--out', type=Path, default=lab/'reports/deployment_4090_20261008/bev_pooling')
    a = p.parse_args()
    budget = json.loads(a.config.read_text())['tensorrt']['acceptance_fixed_before_run']
    results = {}
    for split in ['scene0061', 'mini_val']:
        ref, cand = a.reference_dir/split, a.candidate_dir/split
        rpred, cpred = json.loads((ref/'predictions.json').read_text()), json.loads((cand/'predictions.json').read_text())
        rmetrics = json.loads((a.reference_report/split/'official/metrics/metrics_summary.json').read_text())
        cmetrics = json.loads((a.out/'full_model'/split/'official/metrics/metrics_summary.json').read_text())
        rm, cm = json.loads((ref/'manifest.json').read_text()), json.loads((cand/'manifest.json').read_text())
        provenance = {}
        for name, directory, manifest, evaluation in [
                ('upstream', ref, rm, a.reference_report/split/'official'),
                ('candidate', cand, cm, a.out/'full_model'/split/'official')]:
            provenance[name] = verify_inference_and_evaluation(directory, manifest, evaluation)
        invariants = {key: rm['identity'][key] == cm['identity'][key]
                      for key in ['config', 'packets', 'checkpoint', 'native_library', 'core_library', 'engines', 'onnx']}
        invariants['actual_inputs'] = json.loads((ref/'actual_inputs.json').read_text()) == json.loads((cand/'actual_inputs.json').read_text())
        invariants['pooling_switch'] = rm['identity']['pooling'] == 'upstream' and cm['identity']['pooling'] == 'candidate'
        comparison = compare(rpred, cpred, budget['same_class_match_radius_m'])
        metric_values = {key: dict(upstream=rmetrics[key], candidate=cmetrics[key], delta=cmetrics[key]-rmetrics[key])
                         for key in ['mean_ap', 'nd_score']}
        metrics_pass = (rmetrics['mean_ap'] - cmetrics['mean_ap'] <= budget['map_drop_max'] and
                        rmetrics['nd_score'] - cmetrics['nd_score'] <= budget['nds_drop_max'])
        geometry = {key: comparison['summary'][key] is not None and
                        comparison['summary'][key]['p95'] <= budget['diagnostic_p95_' + key]
                    for key in ['center_m', 'size_m', 'yaw_rad', 'score_abs', 'velocity_mps']}
        comparison.update(invariants=invariants, verified_provenance=provenance, metrics=metric_values, metric_budget_passed=metrics_pass,
                          geometry_diagnostics=geometry, thresholds=budget,
                          class_ap_deltas={cls: cmetrics['mean_dist_aps'][cls]-rmetrics['mean_dist_aps'][cls]
                                           for cls in rmetrics['mean_dist_aps']},
                          status='passed' if all(invariants.values()) and metrics_pass and all(geometry.values()) else 'failed')
        write_json(a.out/'full_model'/split/'comparison.json', comparison)
        rp, cp = json.loads((ref/'performance.json').read_text()), json.loads((cand/'performance.json').read_text())
        results[split] = dict(status=comparison['status'], frames=len(rpred), invariant_checks=invariants, verified_provenance=provenance,
                             metrics=metric_values, geometry_summary=comparison['summary'], geometry_diagnostics=geometry,
                             class_ap_deltas=comparison['class_ap_deltas'],
                             upstream_forward_ms=rp['timings']['forward_h2d_model_d2h_ms'],
                             candidate_forward_ms=cp['timings']['forward_h2d_model_d2h_ms'],
                             upstream_process_peak_mib=rp['process_peak_mib'], candidate_process_peak_mib=cp['process_peak_mib'],
                             matched_boxes=len(comparison['matched']),
                             unmatched_upstream=sum(x['unmatched_reference'] for x in comparison['frames']),
                             unmatched_candidate=sum(x['unmatched_candidate'] for x in comparison['frames']),
                             reference_manifest_sha256=digest(ref/'manifest.json'), candidate_manifest_sha256=digest(cand/'manifest.json'))
    paired_source = a.candidate_dir/'scene0061'
    rows = list(csv.DictReader((paired_source/'full_model_paired_timings.csv').open()))
    pair = json.loads((paired_source/'full_model_paired_summary.json').read_text())
    saving = np.asarray([float(r['upstream_ms']) - float(r['candidate_ms']) for r in rows])
    rng = np.random.default_rng(4090)
    bootstrap = rng.choice(saving, size=(10000, len(saving)), replace=True).mean(axis=1)
    ci = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    pair.update(mean_saved_ms=float(saving.mean()), bootstrap_mean_saved_ms_95ci=ci,
                bootstrap_seed=4090, bootstrap_resamples=10000,
                candidate_faster_pairs=int((saving > 0).sum()),
                upstream_first_mean_saved_ms=float(saving[::2].mean()),
                candidate_first_mean_saved_ms=float(saving[1::2].mean()),
                stable_speedup_supported=ci[0] > 0,
                caveat='Single fixed frame; inspect execution-order effects and do not extrapolate a whole-scene gain.')
    for name in ['full_model_paired_timings.csv', 'full_model_paired_summary.json']:
        (a.out/name).write_bytes((paired_source/name).read_bytes())
    kernel = json.loads((a.out/'results.json').read_text())
    kernel_faster = all(c['timing']['speedup']['p50'] > 1 for c in kernel['cases'] if c['real_capture'])
    full = dict(status='passed' if all(x['status'] == 'passed' for x in results.values()) else 'failed',
                scope='candidate pooling versus identical upstream CUDA/TensorRT model; independent of TensorRT-versus-PyTorch acceptance',
                model_precision='fp16', config_sha256=digest(a.config), thresholds=budget, splits=results, paired=pair,
                stable_speedup_obtained=kernel_faster and pair['stable_speedup_supported'],
                selected_default='upstream', selection_reason=('Isolated pooling does not show consistent gain; keep upstream. ' if not kernel_faster else '') + ('Full-model paired mean-saving CI does not support a positive saving.' if ci[0] <= 0 else 'Single-frame paired saving is positive; broader stable-benefit evidence is still required.'),
                helper_sha256=digest(__file__))
    write_json(a.out/'full_model_comparison.json', full)
    print(json.dumps({key:full[key] for key in ['status', 'stable_speedup_obtained', 'selected_default']}, indent=2))


if __name__ == '__main__':
    main()
