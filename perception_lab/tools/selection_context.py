"""Local pointcloud context used to auto-fit the selected 3D target."""
import json
from pathlib import Path
import numpy as np
from lidarseg_utils import PALETTE
ROOT=Path(__file__).resolve().parents[1]

def context_extent(item):
    return max(3.,float(np.linalg.norm(item['size']))*.75)

def context_mask(points,item):
    return (np.abs(np.asarray(points)-np.asarray(item['center']))<=context_extent(item)).all(axis=1)

def log_context(stream,rr,item,case_id,normal=False):
    packet=json.loads((ROOT/'manifests/frame_packets.json').read_text())[item['frame']]
    sensor=packet['sensors']['LIDAR_TOP'];transform=np.asarray(sensor['T_sensor_to_ego'])
    points=np.fromfile(sensor['path'],dtype=np.float32).reshape(-1,5)[:,:3]
    points=points@transform[:3,:3].T+transform[:3,3]
    keep=context_mask(points,item)
    colors=np.tile([150,160,170],(len(points),1))
    source=ROOT/'outputs/lidarseg/predictions.json'
    if source.exists():
        row=json.loads(source.read_text())[item['frame']]
        if row['sample_token']!=packet['sample_token']:raise ValueError('Pointcloud frame mismatch')
        with np.load(row['path']) as data:labels=data['prediction']
        if labels.shape!=(len(points),) or np.any((labels<0)|(labels>=len(PALETTE))):raise ValueError('Invalid point colors')
        colors=PALETTE[labels]
    root='ego/selection_context/'+case_id
    stream.log(root+'/points',rr.Points3D(points[keep],colors=colors[keep],radii=rr.Radius.ui_points(1),show_labels=False))
    if normal:
        data=json.loads((ROOT/'outputs/rerun/normal_targets.json').read_text())
        boxes=data['frames'][item['frame']][item['module']]
        boxes=[b for b in boxes if np.all(np.abs(np.asarray(b['center_xyz'])-item['center'])+np.linalg.norm(b['size_wlh'])/2<=context_extent(item))]
        if boxes:
            stream.log(root+'/normal',rr.Boxes3D(centers=[b['center_xyz'] for b in boxes],
                sizes=[[b['size_wlh'][1],b['size_wlh'][0],b['size_wlh'][2]] for b in boxes],
                quaternions=[list(b['rotation_wxyz'][1:])+[b['rotation_wxyz'][0]] for b in boxes],
                colors=[100,190,150],show_labels=False))
    return root
