"""Replay fixed-threshold diagnostic cases without changing detection evaluation."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from evidence import sha256
from tracking_overlay import bev_outline,camera_box_segments
COLORS={'false_negative':[255,55,55],'false_positive':[255,65,220],'id_switch':[255,165,20],
        'gap_id_change':[255,165,20],'center_error_over_1m':[255,230,50]}
SHORT={'false_negative':'FN','false_positive':'FP','id_switch':'ID','gap_id_change':'GAP-ID','center_error_over_1m':'ERR>1m'}


def load_cases(directory,packets,score):
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text())
    if not np.isclose(summary['score'],score):raise ValueError('Failure report score differs from viewer --score; use --no-failures or matching report')
    for path,digest in summary['source_sha256'].items():
        if sha256(Path(path))!=digest:raise ValueError('Failure report source changed: '+path)
    rows=json.loads((directory/'cases.json').read_text());by_frame=defaultdict(list);ids=set()
    if dict(Counter(r['kind'] for r in rows))!=summary['case_counts']:raise ValueError('Failure case counts differ from report')
    for row in rows:
        if row['case_id'] in ids or row['sample_token']!=packets[row['frame']]['sample_token']:raise ValueError('Failure case identity/frame mismatch')
        ids.add(row['case_id']);by_frame[row['frame']].append(row)
    return by_frame,summary


def resolve_case(case,detections,tracks,gt_by_instance):
    if case['kind']=='false_positive':
        box=(detections if case['module']=='detection' else tracks)[case['prediction_index']]
    else:box=gt_by_instance[case['instance_token']]
    if box['class_name']!=case['class_name'] or not np.allclose(box['center_xyz'],case['center_ego'],atol=1e-5):raise ValueError('Stale or misaligned failure case')
    return box


def log_failures(rr,rows,detections,tracks,gt_by_instance,points,packet,pose,frame,summary):
    rr.log('failures/detection',rr.Clear(recursive=True));rr.log('failures/tracking',rr.Clear(recursive=True))
    for camera,entry in packet['sensors'].items():
        if camera.startswith('CAM_'):rr.log('ego/cameras/'+camera+'/image/failures',rr.Clear(recursive=True))
    points=points[(np.abs(points[:,0])<55)&(np.abs(points[:,1])<55)&(points[:,2]<3)]
    for module in ['detection','tracking']:
        rr.log('failures/'+module+'/lidar',rr.Points2D(-points[::3,[1,0]],colors=[95,105,115],radii=.025))
    counts=Counter((r['module'],r['kind']) for r in rows)
    text=[f'# Failures | frame {frame}',f"score >= {summary['score']}; center distance < {summary['distance_gate_m']}m. Fixed-threshold diagnostic, NOT official AP matching.",
          'Red=FN; magenta=FP; orange=ID change; yellow=center error >1m. Separate detection/tracking tabs.',
          'Gap recovery with same ID is not highlighted as a failure. Camera projection has no object motion / occlusion correction.']
    for module in ['detection','tracking']:
        text.append(f"\n**{module}**: "+', '.join(f'{SHORT[k]}={counts[(module,k)]}' for k in COLORS))
    bev_groups=defaultdict(list);camera_groups=defaultdict(list)
    for case in rows:
        kind=case['kind']
        if kind not in COLORS:continue
        box=resolve_case(case,detections,tracks,gt_by_instance);color=COLORS[kind]
        label=SHORT[kind]+' '+case['case_id'].replace('case_','')
        if kind in ['id_switch','gap_id_change']:label+=' '+case['previous_tracking_id']+'->'+case['tracking_id']
        bev_groups[(case['module'],kind)].append((box,label,case))
        if case['module']=='detection':
            for camera,entry in packet['sensors'].items():
                if not camera.startswith('CAM_'):continue
                transform=np.linalg.inv(pose)@np.array(entry['T_ego_to_global'])@np.array(entry['T_sensor_to_ego'])
                segments=camera_box_segments(box,transform,entry['K'],entry['width'],entry['height'])
                if segments:
                    xy=np.asarray(segments).reshape(-1,2);anchor=xy[np.argmin(xy[:,1])]
                    camera_groups[(camera,kind)].append((segments,anchor,label,case['case_id']))
        text.append(f"\n{case['case_id']} | {case['module']} | {kind} | {case['class_name']}"+(f" | {case['previous_tracking_id']} -> {case['tracking_id']}" if kind in ['id_switch','gap_id_change'] else ''))
    for (module,kind),items in bev_groups.items():
        rr.log('failures/'+module+'/'+kind,
               rr.LineStrips2D([bev_outline(item[0]) for item in items],colors=COLORS[kind],
                              radii=rr.Radius.ui_points(2),labels=[item[1] for item in items],show_labels=True),
               rr.AnyValues(case_id=[item[2]['case_id'] for item in items],
                            class_name=[item[2]['class_name'] for item in items],
                            sample_token=[item[2]['sample_token'] for item in items]))
    for (camera,kind),items in camera_groups.items():
        path='ego/cameras/'+camera+'/image/failures/'+kind
        rr.log(path+'/boxes',rr.LineStrips2D([segment for item in items for segment in item[0]],
               colors=COLORS[kind],radii=rr.Radius.ui_points(2),draw_order=30))
        rr.log(path+'/labels',rr.Points2D([item[1] for item in items],colors=COLORS[kind],
               labels=[item[2] for item in items],show_labels=True,radii=rr.Radius.ui_points(1),draw_order=31),
               rr.AnyValues(case_id=[item[3] for item in items]))
    rr.log('failure_summary',rr.TextDocument('\n'.join(text),media_type='text/markdown'))
    return {module:{kind:counts[(module,kind)] for kind in COLORS} for module in ['detection','tracking']}
