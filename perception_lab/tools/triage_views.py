"""Failure-only blueprints and a separate, lazily loaded normal target stream."""
import json
import re
from pathlib import Path
import numpy as np
from tracking_overlay import bev_outline,camera_box_segments

APPLICATION_ID='CARperception-triage'
CAMERAS=['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']

def view_paths(module,normal=False,case_id=None):
    if module not in {'detection','tracking'}:raise ValueError('Unknown branch')
    if case_id is not None and not re.fullmatch(r'case_\d{5}',case_id):raise ValueError('Invalid case ID')
    bev=['/failures/'+module+'/lidar','/bev/maptr/**']
    spatial=['/ego/lidar','/ego/maptr/**']
    cameras={}
    for camera in CAMERAS:
        root='/ego/cameras/'+camera+'/image'
        cameras[camera]=[root,root+'/maptr/**']
        if normal:cameras[camera].append(root+'/normal/'+module+'/**')
        if case_id:cameras[camera].append(root+'/selection/'+case_id)
    if case_id:
        bev.append('/selection/'+case_id+'/bev')
        spatial.append('/ego/selection/'+case_id)
    if normal:
        bev.append('/normal/'+module+'/bev/**')
        spatial.append('/ego/normal/'+module+'/**')
    return bev,spatial,cameras

def focus_bounds(points,minimum=96,padding=2.5):
    xy=np.asarray(points,dtype=float).reshape(-1,2)
    if not len(xy) or not np.isfinite(xy).all():raise ValueError('Invalid focus geometry')
    low,high=xy.min(axis=0),xy.max(axis=0)
    center=(low+high)/2;span=np.maximum((high-low)*padding,minimum)
    return [(center[0]-span[0]/2),(center[0]+span[0]/2)],[(center[1]-span[1]/2),(center[1]+span[1]/2)]

def focus_camera(cameras):
    def area(name):
        xy=np.asarray(cameras[name]).reshape(-1,2)
        return float(np.prod(xy.max(axis=0)-xy.min(axis=0)))
    return max(cameras,key=area) if cameras else None

def blueprint(module='detection',normal=False,case_id=None,geometry=None):
    import rerun.blueprint as b
    bev,spatial,cameras=view_paths(module,normal,case_id)
    mode='selected + normal' if normal else 'selected only'
    overview=b.Grid(*[b.Spatial2DView(name=c,origin='/ego/cameras/'+c+'/image',contents=cameras[c]) for c in CAMERAS],grid_columns=3)
    bev_bounds=None
    right=overview
    eye_options={}
    if geometry is not None:
        from selection_context import camera_eye
        position,target=camera_eye(geometry)
        eye_options={'eye_controls':b.EyeControls3D(position=position,look_target=target,eye_up=[0,0,1])}
        x,y=focus_bounds(geometry['bev'],minimum=12,padding=3)
        bev_bounds=b.VisualBounds2D(x_range=x,y_range=y)
        camera=focus_camera(geometry['cameras'])
        if camera:
            x,y=focus_bounds(geometry['cameras'][camera])
            detail=b.Spatial2DView(name='Target close-up / '+case_id+' / '+camera,
                origin='/ego/cameras/'+camera+'/image',contents=cameras[camera],
                visual_bounds=b.VisualBounds2D(x_range=x,y_range=y))
            right=b.Vertical(detail,overview,row_shares=[.6,.4])
    return b.Blueprint(b.Horizontal(
        b.Vertical(b.Spatial2DView(name=module+' / BEV / '+mode,origin='/',contents=bev,visual_bounds=bev_bounds),
                   b.Spatial3DView(name=module+(' / 3D close-up / ' if case_id else ' / 3D / ')+mode,origin='/ego',contents=spatial,**eye_options),
                   b.TextDocumentView(name='Selected case' if case_id else 'Frame diagnostic summary',origin='/selection/'+case_id+'/summary' if case_id else '/failure_summary'),
                   row_shares=[.42,.35,.23]),
        right,column_shares=[.45,.55]),b.TimePanel(state='expanded'),collapse_panels=True)

def save_extras(directory,recording_id,packets,normal_data):
    import rerun as rr
    from rerun_mini import box_components
    directory=Path(directory)
    for module in ['detection','tracking']:
        for show in [False,True]:
            stream=rr.RecordingStream(APPLICATION_ID,recording_id=recording_id,send_properties=False)
            stream.save(directory/('view_'+module+('_normal' if show else '_errors')+'.rrd'))
            rr.send_blueprint(blueprint(module,show),recording=stream,make_active=True,make_default=True)
            stream.flush();stream.disconnect()
    stream=rr.RecordingStream(APPLICATION_ID,recording_id=recording_id,send_properties=False)
    stream.save(directory/'normal_targets.rrd')
    color=[100,190,150]
    for i,packet in enumerate(packets):
        stream.set_time('frame',sequence=i)
        stream.set_time('elapsed',duration=(packet['timestamp_us']-packets[0]['timestamp_us'])/1e6)
        pose=np.asarray(packet['sensors']['LIDAR_TOP']['T_ego_to_global'])
        for module in ['detection','tracking']:
            boxes=normal_data['frames'][i][module]
            paths=['normal/'+module+'/bev','ego/normal/'+module]
            paths += ['ego/cameras/'+c+'/image/normal/'+module for c in CAMERAS]
            for path in paths:stream.log(path,rr.Clear(recursive=True))
            if not boxes:continue
            labels=['OK '+str(b.get('tracking_id',b['class_name'])) for b in boxes]
            centers,sizes,quats=box_components(boxes)
            stream.log('ego/normal/'+module+'/boxes',rr.Boxes3D(centers=centers,sizes=sizes,quaternions=quats,
                colors=color,labels=labels,show_labels=False))
            stream.log('normal/'+module+'/bev/boxes',rr.LineStrips2D([bev_outline(b) for b in boxes],
                colors=color,radii=rr.Radius.ui_points(1),labels=labels,show_labels=False))
            for camera in CAMERAS:
                entry=packet['sensors'][camera]
                transform=np.linalg.inv(pose)@np.array(entry['T_ego_to_global'])@np.array(entry['T_sensor_to_ego'])
                segments=[s for box in boxes for s in camera_box_segments(box,transform,entry['K'],entry['width'],entry['height'])]
                if segments:stream.log('ego/cameras/'+camera+'/image/normal/'+module+'/boxes',
                    rr.LineStrips2D(segments,colors=color,radii=rr.Radius.ui_points(1),draw_order=15))
    stream.set_time('normal_ready',sequence=1)
    stream.log('normal_ready',rr.TextLog('Normal targets fully loaded'))
    stream.flush();stream.disconnect()
