"""Write a small selection blueprint for an existing recording (no scene reload)."""
import argparse,json
from pathlib import Path
import rerun as rr
from triage_views import APPLICATION_ID,blueprint

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--recording-id',required=True);p.add_argument('--module',choices=['detection','tracking'],required=True)
    p.add_argument('--normal',action='store_true');p.add_argument('--case-id',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    stream=rr.RecordingStream(APPLICATION_ID,recording_id=a.recording_id,send_properties=False)
    stream.save(a.out)
    geometry=json.loads((Path(__file__).resolve().parents[1]/'outputs/rerun/selection_geometry.json').read_text())
    if geometry['recording_id']!=a.recording_id:raise ValueError('Selection geometry recording mismatch')
    item=geometry['cases'][a.case_id]
    if item['module']!=a.module:raise ValueError('Selection branch mismatch')
    color=[160,255,255]
    paths=['selection/'+a.case_id+'/bev','ego/selection/'+a.case_id]
    paths+=['ego/cameras/'+camera+'/image/selection/'+a.case_id for camera in item['cameras']]
    stream.set_time('frame',sequence=item['frame'])
    stream.log(paths[0],rr.LineStrips2D([item['bev']],colors=color,radii=rr.Radius.ui_points(3),draw_order=50),rr.AnyValues(case_id=a.case_id))
    stream.log(paths[1],rr.Boxes3D(centers=[item['center']],sizes=[item['size']],quaternions=[item['quaternion']],
        colors=color,radii=rr.Radius.ui_points(3),show_labels=False),rr.AnyValues(case_id=a.case_id))
    for camera,segments in item['cameras'].items():
        stream.log('ego/cameras/'+camera+'/image/selection/'+a.case_id,
            rr.LineStrips2D(segments,colors=color,radii=rr.Radius.ui_points(3),draw_order=50),rr.AnyValues(case_id=a.case_id))
    if item['frame']+1<geometry['frame_count']:
        stream.set_time('frame',sequence=item['frame']+1)
        for path in paths:stream.log(path,rr.Clear(recursive=True))
    rr.send_blueprint(blueprint(a.module,a.normal,a.case_id,item),recording=stream,make_active=True,make_default=True)
    stream.flush();stream.disconnect()
