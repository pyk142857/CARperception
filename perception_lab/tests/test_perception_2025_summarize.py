"""A stale metric file must not manufacture a newly successful experiment."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import perception_2025_summarize as summary


def test_failed_or_mismatched_native_metrics_are_excluded(tmp_path):
    folder = tmp_path/'deimv2'
    summary.write(folder/'official_eval/metrics.json', {'AP': .5})
    summary.write(folder/'official_eval/protocol.json', {'images': 2})
    summary.write(folder/'image_ids.json', [1, 2])
    summary.write(folder/'status.json', {'status': 'failed', 'coco_images': 2})
    assert not summary.collect(tmp_path)[1]
    summary.write(folder/'status.json', {'status': 'passed', 'coco_images': 5000})
    assert not summary.collect(tmp_path)[1]
    summary.write(folder/'status.json', {'status': 'partial', 'coco_images': 2})
    assert summary.collect(tmp_path)[1][0]['samples'] == 2


def test_empty_experiment_report_does_not_claim_completed_models(tmp_path, monkeypatch):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'report': str(tmp_path/'report'), 'models': {}}))
    monkeypatch.setattr(sys, 'argv', ['summarize', '--config', str(config)])
    summary.main()
    report = (tmp_path/'report/report.md').read_text()
    assert '完成 COCO 5,000' not in report
    assert 'AMOTA0.86257' not in report
    assert json.loads((tmp_path/'report/metrics.json').read_text()) == []
    assert json.loads((tmp_path/'report/summarize/status.json').read_text())['status'] == 'passed'


def test_grae_requires_official_evaluation_and_consistent_saved_outputs(tmp_path):
    folder = tmp_path/'grae3dmot'
    row = dict(input='frozen_reference', split='mini_val', frames=1, amota=.7, amotp=.5, ids=1, fp=2, fn=3)
    summary.write(folder/'metrics.json', [row])
    summary.write(folder/'status.json', dict(status='partial', inference='passed', official_evaluation='pending'))
    assert not summary.collect(tmp_path)[1]
    summary.write(folder/'status.json', dict(status='partial', inference='passed', official_evaluation='passed'))
    summary.write(folder/'frozen_reference/mini_val/tracking.json', {'results': {'frame': []}})
    summary.write(folder/'frozen_reference/mini_val/official_eval/metrics_summary.json', row)
    assert len(summary.collect(tmp_path)[1]) == 5
    row['ids'] = 9
    summary.write(folder/'metrics.json', [row])
    assert not summary.collect(tmp_path)[1]
