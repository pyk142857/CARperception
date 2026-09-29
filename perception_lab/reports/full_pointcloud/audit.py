"""Check every displayed full-cloud frame against raw LiDAR and calibration."""
import json,hashlib
from pathlib import Path
import numpy as np
import rerun.dataframe as rdf
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
packets=json.loads((ROOT/'manifests/frame_packets.json').read_text())
recording=ROOT/'outputs/rerun/triage_scene.rrd'
table=rdf.load_recording(recording).view(index='frame',contents='/ego/lidar').select().read_all()
assert table.num_rows==len(packets)==39
counts=[];error=0.
for row in table.select(['frame','/ego/lidar:Points3D:positions']).to_pylist():
 frame=row['frame'];sensor=packets[frame]['sensors']['LIDAR_TOP']
 raw=np.fromfile(sensor['path'],dtype=np.float32).reshape(-1,5)
 transform=np.asarray(sensor['T_sensor_to_ego']);expected=raw[:,:3]@transform[:3,:3].T+transform[:3,3]
 actual=np.asarray(row['/ego/lidar:Points3D:positions'])
 assert actual.shape==expected.shape,(frame,actual.shape,expected.shape)
 delta=float(np.max(np.abs(actual-expected)));assert delta<1e-4
 counts.append(dict(frame=frame,raw_points=len(raw),recorded_points=len(actual)));error=max(error,delta)
(HERE/'pointcloud_audit.json').write_text(json.dumps(dict(frames=counts,total_points=sum(c['raw_points'] for c in counts),max_coordinate_error_m=error,recording_sha256=hashlib.sha256(recording.read_bytes()).hexdigest()),indent=2)+'\n')
print('All 39 frames retain every raw LiDAR point:',sum(c['raw_points'] for c in counts))
