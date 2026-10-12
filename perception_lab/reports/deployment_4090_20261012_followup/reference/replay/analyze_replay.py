"""CPU summary of completed identical-input module and mapping replays."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from bevfusion_reference_followup import tensor_comparison
OUT = Path(__file__).parent
summary = json.loads((OUT / 'summary.json').read_text())
rows, aligned = [], []
for case in summary['cases']:
    token, name = case['token'], case['case']
    for node, result in case['nodes'].items():
        rows.append({'sample_token': token, 'case': name, 'node': node,
            'scope': 'identical input local module; head rowwise not source aligned',
            'passed': result['passed'], 'exceeded_elements': result['exceeded_elements'],
            'max_abs': result['absolute_error']['max'], 'p95_abs': result['absolute_error']['p95']})
    if 'geometry_mapping_replay' in case:
        result = case['geometry_mapping_replay']['mapping_only_difference']
        rows.append({'sample_token': token, 'case': name, 'node': 'pool_mapping_only',
            'scope': 'same products, same FP32 kernel, canonical source order, different actual mappings',
            'passed': result['passed'], 'exceeded_elements': result['exceeded_elements'],
            'max_abs': result['absolute_error']['max'], 'p95_abs': result['absolute_error']['p95']})
    if not name.startswith('head.'):
        continue
    raw = ROOT / 'runs/deployment_4090_20261012_followup'
    reference = np.load(raw / 'reference/replay' / token / (name + '.outputs.npz'))
    marked = np.load(raw / 'trt/replay' / token / (name + '.marked.npz'))
    ref_ids = np.asarray(case['reference_query_source_indices']).ravel()
    trt_ids = np.asarray(case['marked_trt_query_source_indices']).ravel()
    common, ri, ti = np.intersect1d(ref_ids, trt_ids, return_indices=True)
    item = {'sample_token': token, 'case': name, 'common_source_count': len(common),
        'reference_only_sources': np.setdiff1d(ref_ids, trt_ids).tolist(),
        'marked_trt_only_sources': np.setdiff1d(trt_ids, ref_ids).tolist(),
        'marked_outputs_bitwise_equal_production': case['marked_trt_outputs_equal_production'],
        'scope': 'actual seed identity alignment to marked diagnostic graph, not proven production provenance',
        'nodes': {}}
    for node in reference.files:
        result = tensor_comparison(reference[node][0][:, ri], marked[node][0][:, ti])
        item['nodes'][node] = result
        rows.append({'sample_token': token, 'case': name, 'node': 'source_aligned.' + node,
            'scope': item['scope'], 'passed': result['passed'], 'exceeded_elements': result['exceeded_elements'],
            'max_abs': result['absolute_error']['max'], 'p95_abs': result['absolute_error']['p95']})
    aligned.append(item)
(OUT / 'source_aligned_head.json').write_text(json.dumps(aligned, indent=2) + '\n')
with (OUT / 'metrics.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
print('Rows:', len(rows), 'head cases:', len(aligned))
