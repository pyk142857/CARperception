"""Isolated frozen-detector / GRAE / OptiPMB comparison; CPU export only."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import numpy as np
from pyquaternion import Quaternion
from perception_2025_visualize import (ROOT, sha, read_json, ego_boxes, transform_points,
    camera_to_reference, box_components, track_color, bev_outline, camera_box_segments, CAMERAS)
from deployment_visualize import render

REPORT=ROOT/'reports/tracking_gpu_20261013'
OUTPUT=ROOT/'outputs/rerun/deployment_4090/tracking_gpu_20261013'
FAMILIES=('bev_detection','grae','optipmb')


def load_inputs(split, report=REPORT):
    manifest=read_json(report/'input_manifest.json')
    if manifest.get('status')!='passed':raise ValueError('Frozen input audit not passed')
    entry=next(x for x in manifest['splits'] if x['split']==split)
    packet_path=ROOT/entry['packets_path'];detection_path=ROOT/entry['detection_path']
    if sha(packet_path)!=entry['packets_sha256'] or sha(detection_path)!=entry['detection_sha256']:
        raise ValueError('Frozen source hash mismatch')
    packets=read_json(packet_path);tokens=[p['sample_token'] for p in packets]
    assert len(tokens)==entry['frames'] and len(set(tokens))==len(tokens)
    assert tokens==[s['sample_token'] for s in entry['samples']]
    artifacts={'bev_detection':read_json(detection_path)['results']}
    hashes={'input_manifest':sha(report/'input_manifest.json'),'packets':sha(packet_path),'bev_detection':sha(detection_path)}
    for family in ('grae','optipmb'):
        base=report/'native_compare'/family
        folder=base/'frozen_reference' if family=='grae' else base
        tracking=folder/split/'tracking.json';protocol_path=folder/split/'official_eval/protocol.json'
        protocol=read_json(protocol_path);digest=sha(tracking)
        if family=='grae':
            state=read_json(folder/'status.json');receipt=read_json(folder/'native_receipt.json')
            assert state.get('status') in ('partial','passed') and state.get('inference')=='passed' and state.get('official_evaluation')=='passed'
            assert receipt['tracking_sha256'][split]==digest
        else:
            state=read_json(base/'evaluation_status.json');validation=read_json(base/'validation_summary.json')
            assert state.get('status')=='passed' and validation[split]['tracking_file_sha256']==digest
            row=next(r for r in state['rows'] if r['split']==split)
            assert row['tracking_sha256']==digest and row['metrics_sha256']==sha(folder/split/'official_eval/metrics_summary.json')
        evaluation=read_json(folder/'evaluation_status.json')
        assert evaluation.get('status')=='passed'
        evaluated=next(r for r in evaluation['rows'] if r['split']==split)
        assert evaluated['tracking_sha256']==digest and evaluated['metrics_sha256']==sha(folder/split/'official_eval/metrics_summary.json')
        for name,expected in protocol.get('gt_sha256',{}).items():
            assert sha(ROOT/'data/nuscenes/v1.0-mini'/name)==expected
        assert protocol['prediction_sha256']==digest
        assert protocol['frozen_detection_sha256']==hashes['bev_detection'] and protocol['packets_sha256']==hashes['packets']
        assert protocol['tokens']==tokens
        artifacts[family]=read_json(tracking)['results'];hashes[family]=digest;hashes[family+'_protocol']=sha(protocol_path)
    for family,data in artifacts.items():
        if set(data)!=set(tokens):raise ValueError('Model token mismatch: '+family)
    for packet,expected in zip(packets,entry['samples']):
        lidar=packet['sensors']['LIDAR_TOP']
        if sha(lidar['path'])!=expected['raw_lidar_sha256']:raise ValueError('Raw LiDAR hash mismatch')
        matrix=np.asarray(lidar['T_ego_to_global'])@np.asarray(lidar['T_sensor_to_ego'])
        if not np.allclose(matrix,expected['sensor_to_global'],atol=1e-10):raise ValueError('LiDAR pose mismatch')
    return packets,artifacts,hashes


def decoded_audit(path,packets,counts,artifacts):
    import rerun.dataframe as rdf
    recording=rdf.load_recording(str(path))
    def values(entity,component):
        t=recording.view(index='frame',contents=entity,include_tombstone_columns=True).select().read_all()
        return dict(zip(t['frame'].to_pylist(),t['/'+entity+':'+component].to_pylist()))
    frames=list(range(len(packets)));points=values('sensors/lidar','Position3D');error=0.;box_error=0.
    assert list(points)==frames
    for i,p in enumerate(packets):
        lidar=p['sensors']['LIDAR_TOP'];raw=np.fromfile(lidar['path'],dtype=np.float32).reshape(-1,5)[:,:3]
        expected=transform_points(raw,np.asarray(lidar['T_sensor_to_ego']));actual=np.asarray(points[i])
        assert actual.shape==expected.shape and len(actual)==counts[i]['original_lidar_points']
        error=max(error,float(np.max(np.abs(actual-expected))))
    assert error<1e-4
    assert [v[0] for v in values('sample','Text').values()]==[p['sample_token'] for p in packets]
    for entity in ['models','bev/models','ground_truth','bev/ground_truth']+['cameras/'+c for c in CAMERAS]:
        clear=values(entity,'ClearIsRecursive');assert list(clear)==frames and all(v==[True] for v in clear.values())
    totals={}
    for family in FAMILIES:
        actual=values('models/'+family+'/boxes','PoseTranslation3D')
        for i,p in enumerate(packets):
            pose=np.asarray(p['sensors']['LIDAR_TOP']['T_ego_to_global'])
            boxes=ego_boxes(artifacts[family][p['sample_token']],pose,family)
            assert len(actual.get(i) or [])==counts[i]['model_boxes'][family]==len(boxes)
            if boxes:box_error=max(box_error,float(np.max(np.abs(np.asarray(actual[i])-np.asarray([b['center_xyz'] for b in boxes])))))
        totals[family]=sum(c['model_boxes'][family] for c in counts)
    gt=values('ground_truth/boxes','PoseTranslation3D')
    assert all(len(gt.get(i) or [])==row['gt_boxes'] for i,row in enumerate(counts))
    projected={}
    for cam in CAMERAS:
        for family in (*FAMILIES,'ground_truth'):
            expected=[row['camera_segments'][cam][family] for row in counts]
            if not any(expected):continue
            segments=values('cameras/'+cam+'/image/'+family+'/boxes','LineStrip2D')
            assert [len(segments.get(i) or []) for i in frames]==expected
            projected[cam+'/'+family]=sum(expected)
    assert box_error<1e-4
    return dict(status='passed',decoded_frames=len(packets),decoded_points=sum(r['original_lidar_points'] for r in counts),
        decoded_box_counts=totals,decoded_camera_segments=projected,float32_point_max_abs_m=error,float32_box_center_max_abs_m=box_error,
        token_order_exact=True,recursive_clear_frames=len(frames),representative_frames=[0,len(frames)//2,len(frames)-1])


def export(split):
    suffix='' if split=='scene0061' else '/mini_val'
    report=REPORT/('visualization'+suffix);report.mkdir(parents=True,exist_ok=True)
    out=Path(str(OUTPUT)+suffix);out.mkdir(parents=True,exist_ok=True)
    status_path=report/'status.json'
    status_path.write_text(json.dumps(dict(status='running',export='running',browser_validation='pending',split=split)))
    packets,artifacts,hashes=load_inputs(split)
    gt={}
    for row in read_json(ROOT/'data/nuscenes/v1.0-mini/sample_annotation.json'):gt.setdefault(row['sample_token'],[]).append(row)
    import rerun as rr
    import rerun.blueprint as rrb
    rr.init('CAR-tracking-gpu-'+split,strict=True);rr.save(str(out/'tracking_gpu.building.rrd'))
    rr.log('/',rr.ViewCoordinates.FLU,static=True)
    tabs=[]
    for family in (*FAMILIES,'ground_truth'):
        layer='ground_truth' if family=='ground_truth' else 'models/'+family
        tabs.append(rrb.Horizontal(rrb.Vertical(rrb.Spatial3DView(name=family,origin='/',contents=['+ /'+layer+'/**','+ /sensors/lidar']),
            rrb.Spatial2DView(name='BEV '+family,origin='bev',contents=['+ /bev/'+layer+'/**','+ /bev/lidar'])),
            rrb.Grid(*[rrb.Spatial2DView(name=cam,origin='cameras/'+cam+'/image',contents=['+ /cameras/'+cam+'/image','+ /cameras/'+cam+'/image/'+family+'/**']) for cam in CAMERAS],grid_columns=3),name=family))
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(rrb.Tabs(*tabs,active_tab=0),rrb.TextDocumentView(origin='guide'),row_shares=[9,1]),collapse_panels=True))
    counts=[];history={};last_scene=None;gaps=0
    for i,p in enumerate(packets):
        if p['scene_token']!=last_scene:history={};last_scene=p['scene_token']
        rr.set_time('frame',sequence=i);rr.set_time('elapsed',duration=(p['timestamp_us']-packets[0]['timestamp_us'])/1e6)
        token=p['sample_token'];rr.log('sample',rr.TextDocument(token))
        lidar=p['sensors']['LIDAR_TOP'];pose=np.asarray(lidar['T_ego_to_global']);inverse=np.linalg.inv(pose)
        cloud=transform_points(np.fromfile(lidar['path'],dtype=np.float32).reshape(-1,5)[:,:3],np.asarray(lidar['T_sensor_to_ego']))
        assert np.isfinite(cloud).all()
        rr.log('sensors/lidar',rr.Points3D(cloud,colors=[110,130,145],radii=.03));rr.log('bev/lidar',rr.Points2D(-cloud[:,[1,0]],colors=[70,90,110],radii=.04))
        for entity in ['models','bev/models','ground_truth','bev/ground_truth']:rr.log(entity,rr.Clear(recursive=True))
        boxes={family:ego_boxes(data[token],pose,family) for family,data in artifacts.items()}
        for family,items in boxes.items():
            if not items:continue
            centers,sizes,rots=box_components(items);colors=[track_color(b['track_id']) for b in items]
            rr.log('models/'+family+'/boxes',rr.Boxes3D(centers=centers,sizes=sizes,quaternions=rots,colors=colors,labels=[b['track_id'] for b in items],show_labels=False))
            rr.log('bev/models/'+family+'/boxes',rr.LineStrips2D([bev_outline(b) for b in items],colors=colors))
            if family=='bev_detection':continue
            for b in items:
                ident=b['track_id'];old=history.get(ident,[])
                if old and old[-1][0]!=i-1:old=[];gaps+=1
                old=(old+[(i,transform_points(np.asarray([b['center_xyz']]),pose)[0])])[-12:];history[ident]=old
                if len(old)>1:
                    trail=transform_points(np.asarray([v[1] for v in old]),inverse)
                    rr.log('models/'+family+'/tracks/'+ident,rr.LineStrips3D([trail],colors=track_color(ident)))
                    rr.log('bev/models/'+family+'/tracks/'+ident,rr.LineStrips2D([-trail[:,[1,0]]],colors=track_color(ident)))
        q=Quaternion(matrix=pose[:3,:3])
        gtboxes=[dict(center_xyz=transform_points(np.asarray([b['translation']]),inverse)[0],size_wlh=b['size'],rotation_wxyz=(q.inverse*Quaternion(b['rotation'])).elements) for b in gt.get(token,[])]
        if gtboxes:
            centers,sizes,rots=box_components(gtboxes);rr.log('ground_truth/boxes',rr.Boxes3D(centers=centers,sizes=sizes,quaternions=rots,colors=[60,230,120]))
            rr.log('bev/ground_truth/boxes',rr.LineStrips2D([bev_outline(b) for b in gtboxes],colors=[60,230,120]))
        camera_segments={}
        for cam in CAMERAS:
            e=p['sensors'][cam];tr=camera_to_reference(e,pose);path='cameras/'+cam
            rr.log(path,rr.Clear(recursive=True));rr.log(path,rr.Transform3D(translation=tr[:3,3],mat3x3=tr[:3,:3]))
            rr.log(path+'/image',rr.Pinhole(image_from_camera=e['K'],width=e['width'],height=e['height'],camera_xyz=rr.ViewCoordinates.RDF))
            rr.log(path+'/image',rr.EncodedImage(path=e['path']))
            camera_segments[cam]={}
            for family,items in {**boxes,'ground_truth':gtboxes}.items():
                segs=[];colors=[]
                for b in items:
                    seg=camera_box_segments(b,tr,e['K'],e['width'],e['height']);segs+=seg
                    colors += [track_color(b['track_id']) if 'track_id' in b else [60,230,120]]*len(seg)
                camera_segments[cam][family]=len(segs)
                if segs:rr.log(path+'/image/'+family+'/boxes',rr.LineStrips2D(segs,colors=colors))
        row=dict(frame=i,sample_token=token,scene_token=p['scene_token'],timestamp_us=p['timestamp_us'],original_lidar_points=len(cloud),gt_boxes=len(gtboxes),model_boxes={k:len(v) for k,v in boxes.items()},camera_segments=camera_segments,missing_models=[])
        counts.append(row)
        rr.log('guide',rr.TextDocument('Frozen BEVFusion detections / current native GRAE / OptiPMB / independent GT. Display >= .25, evaluation unfiltered. Full raw cloud; asynchronous camera poses; no model input uses GT.\n'+json.dumps(row)))
        im=render(p,'OptiPMB / frozen BEVFusion native detections',boxes['optipmb'],cloud,None,[],i,{})
        from PIL import ImageDraw
        draw=ImageDraw.Draw(im);draw.rectangle((1340,15,1919,55),fill=(8,14,22));draw.text((1340,22),f'frame {i:02d}/{len(packets)-1} | measured',fill=(180,205,220))
        im.save(out/f'frame_{i:02d}.jpg',quality=86)
        if i in [0,len(packets)//2,len(packets)-1]:im.save(report/f'frame_{i:02d}.jpg',quality=90)
    rr.get_global_data_recording().flush();rr.disconnect();(out/'tracking_gpu.building.rrd').replace(out/'tracking_gpu.rrd')
    audit=decoded_audit(out/'tracking_gpu.rrd',packets,counts,artifacts)
    url='http://127.0.0.1:9094/tracking_gpu_20261013'+suffix+'/'
    rerun_url='http://127.0.0.1:9095/?url='+quote(url+'tracking_gpu.rrd',safe='')+'&renderer=webgl&persist=0'
    (out/'data.json').write_text(json.dumps(counts))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font-family:sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}</style><title>Tracking comparison</title><h1>Frozen BEVFusion / GRAE / OptiPMB</h1><p>Below: OptiPMB snapshot. Choose detector, tracker, or independent GT tabs in Rerun.</p><a href="'+rerun_url+'">Open Rerun comparison</a><input id="frame" type="range" min="0" max="'+str(len(packets)-1)+'" value="0"><pre id="info"></pre><img id="image"><script>fetch("data.json").then(r=>r.json()).then(d=>{function update(){image.src="frame_"+String(frame.value).padStart(2,"0")+".jpg";info.textContent=JSON.stringify(d[+frame.value],null,2)}frame.oninput=update;update()})</script>')
    summary=dict(status='partial',export='passed',browser_validation='pending',split=split,frames=len(packets),counts=counts,artifact_sha256=hashes,rrd=str(out/'tracking_gpu.rrd'),rrd_sha256=sha(out/'tracking_gpu.rrd'),url=url,rerun_url=rerun_url,decoded_rrd_audit=audit,missing_models=[],trajectory_gap_resets=gaps,generated_at=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(summary,indent=2)+'\n');(report/'decoded_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['scene0061','mini_val'],default='scene0061')
    export(parser.parse_args().split)
