"""CPU evidence: do changed depth pixels select different values from identical scatter inputs?"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'runs/deployment_4090_20261012_followup/reference'
OUT = Path(__file__).parent / 'raster_collision_evidence.json'
rows = []
for folder in sorted(RAW.iterdir()):
    if not (folder / 'repeat_04.npz').exists():
        continue
    with np.load(folder / 'repeat_00.npz') as baseline:
        depth0 = baseline['camera.sparse_depth']
        for repeat in range(1, 5):
            with np.load(folder / ('repeat_%02d.npz' % repeat)) as candidate:
                depth1 = candidate['camera.sparse_depth']
                changed = np.argwhere(depth0 != depth1)
                identical_scatter_inputs = all(np.array_equal(baseline[key], candidate[key])
                    for key in baseline.files if key.startswith('camera.raster.'))
                collision_count = membership_count = 0
                examples = []
                for camera in range(6):
                    prefix = 'camera.raster.' + str(camera)
                    scatter_rows = baseline[prefix + '.rows']
                    scatter_cols = baseline[prefix + '.cols']
                    values = baseline[prefix + '.depth_values']
                    for _, _, row, col in changed[changed[:, 0] == camera]:
                        contributions = values[(scatter_rows == row) & (scatter_cols == col)]
                        old, new = depth0[camera, 0, row, col], depth1[camera, 0, row, col]
                        collision_count += int(len(contributions) > 1)
                        membership = bool(np.any(contributions == old) and np.any(contributions == new))
                        membership_count += int(membership)
                        if len(examples) < 20:
                            examples.append({'camera_index': camera, 'row': int(row), 'col': int(col),
                                'reference_depth': float(old), 'candidate_depth': float(new),
                                'actual_scatter_contribution_depths': contributions.tolist(),
                                'both_outputs_are_actual_contributions': membership})
                rows.append({'token': folder.name, 'repeat': repeat,
                    'actual_scatter_arguments_bitwise_equal': identical_scatter_inputs,
                    'changed_sparse_depth_pixels': len(changed),
                    'changed_pixels_with_multiple_contributions': collision_count,
                    'changed_pixels_where_both_outputs_are_actual_contributions': membership_count,
                    'examples': examples})
OUT.write_text(json.dumps({'scope': 'instrumented reference only; observations do not quantify TRT precision contribution',
    'method': 'exact value comparison of captured scatter arguments and sparse-depth output; repeat0 baseline',
    'comparisons': rows}, indent=2) + '\n')
print(OUT)

# Candidate-source alignment, with ordinary 2pi yaw wrapping (no class symmetry relaxation).
import csv
report_dir = Path(__file__).parent
summary = json.loads((report_dir / 'summary.json').read_text())
metrics = []
for token_entry in summary['tokens']:
    token = token_entry['sample_token']
    report = json.loads((report_dir / (token + '.json')).read_text())
    baseline = {b['query_source_index']: b for b in report['repeat'][0]['native_boxes']}
    errors = []
    for repeat in report['repeat'][1:]:
        candidate = {b['query_source_index']: b for b in repeat['native_boxes']}
        for source in baseline.keys() & candidate.keys():
            a, b = baseline[source], candidate[source]
            delta = np.array(b['tensor']) - a['tensor']
            errors.append([np.linalg.norm(delta[:3]), np.max(np.abs(delta[3:6])),
                abs(np.arctan2(np.sin(delta[6]), np.cos(delta[6]))), np.linalg.norm(delta[7:9]),
                abs(a['score'] - b['score'])])
    errors = np.asarray(errors)
    collisions = [r for r in rows if r['token'] == token]
    comparisons = report['comparisons_to_repeat0']
    row = {'sample_token': token, 'compared_repeat_pairs': 4, 'source_aligned_pairs': len(errors),
        'changed_sparse_depth_pixels_min': min(r['changed_sparse_depth_pixels'] for r in collisions),
        'changed_sparse_depth_pixels_max': max(r['changed_sparse_depth_pixels'] for r in collisions),
        'sparse_depth_max_abs': max(c['nodes']['camera.sparse_depth']['absolute_error']['max'] for c in comparisons),
        'source_removed_max': max(len(c['native_boxes_by_source']['removed_sources']) for c in comparisons),
        'source_added_max': max(len(c['native_boxes_by_source']['added_sources']) for c in comparisons),
        'source_reordered_pairs': sum(not c['native_boxes_by_source']['same_order'] for c in comparisons)}
    for index, field in enumerate(['center_m', 'size_m', 'wrapped_yaw_rad', 'velocity_mps', 'score_abs']):
        row[field + '_max'] = float(errors[:, index].max())
        row[field + '_p95'] = float(np.percentile(errors[:, index], 95))
    metrics.append(row)
with (report_dir / 'metrics.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=metrics[0])
    writer.writeheader()
    writer.writerows(metrics)
(report_dir / 'source_aligned_metrics.json').write_text(json.dumps({
    'scope': 'Instrumented same-backend repeats versus repeat0; actual class*HW query seed identity; '
             'yaw wraps 2pi, no pi symmetry relaxation', 'tokens': metrics}, indent=2) + '\n')
