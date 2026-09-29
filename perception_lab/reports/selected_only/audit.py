"""Verify the main recording contains background, not preloaded failure boxes."""
import subprocess,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
recording=ROOT/'outputs/rerun/triage_scene.rrd'
text=subprocess.check_output([str(ROOT/'envs/rerun/bin/rerun'),'rrd','print',str(recording)],text=True)
for token in ['false_positive','false_negative','id_switch','gap_id_change','center_error_over_1m','Boxes3DIndicator']:
    assert token not in text,token
assert '/failures/detection/lidar' in text and '/failures/tracking/lidar' in text
metadata=json.loads(recording.with_suffix('.json').read_text())
(HERE/'audit.json').write_text(json.dumps(dict(recording_id=metadata['recording_id'],
    recording_sha256=hashlib.sha256(recording.read_bytes()).hexdigest(),
    cases_sha256=metadata['failure_report_sha256'],frame_count=metadata['frame_count'],
    check='No failure-kind geometry entities or Boxes3D components in main recording; BEV pointcloud backgrounds retained'),indent=2)+'\n')
print('Main recording contains no preloaded failure boxes')
