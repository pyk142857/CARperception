"""Aggregate existing diagnostic observations; no driving-risk labels inferred."""
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'mini_evaluation/cases.json'
REGIONS = ['front_near', 'front_sides', 'rear_near', 'front_far', 'other']
KINDS = ['false_negative', 'false_positive', 'id_switch', 'gap_id_change',
         'center_error_over_1m']


def region(x, y):
    if 0 <= x <= 30 and abs(y) <= 3:
        return 'front_near'
    if 0 <= x <= 30 and 3 < abs(y) <= 15:
        return 'front_sides'
    if -20 <= x < 0 and abs(y) <= 6:
        return 'rear_near'
    if 30 < x <= 50 and abs(y) <= 6:
        return 'front_far'
    return 'other'


def main():
    cases = json.loads(SOURCE.read_text())
    failures = [c for c in cases if c['kind'] in KINDS]
    counts = collections.Counter((c['module'], region(*c['center_ego'][:2]),
                                  c['kind']) for c in failures)
    assert len(failures) == 1910
    assert sum(counts.values()) == len(failures)
    with (HERE / 'region_counts.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['module', 'region', *KINDS])
        for module in ['detection', 'tracking']:
            for area in REGIONS:
                writer.writerow([module, area, *[counts[module, area, k] for k in KINDS]])
    examples = [c for c in failures if c['module'] == 'detection'
                and c['kind'] == 'false_negative'
                and region(*c['center_ego'][:2]) == 'front_near']
    (HERE / 'summary.json').write_text(json.dumps({
        'source': str(SOURCE.relative_to(HERE.parents[2])),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'frames': 39, 'score_threshold': 0.25, 'match_distance_m': 2.0,
        'unit': 'object-frame diagnostic observations, not unique objects',
        'excluded_kind': 'gap_recovery', 'failure_observations': len(failures),
        'front_near_detection_fn': examples,
        'limitations': 'Rectangular screening regions, not lanes or collision envelopes; '
                       'no planner, control or driving-impact validation.'
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'Partitioned {len(failures)} failure observations into five disjoint regions.')


if __name__ == '__main__':
    main()
