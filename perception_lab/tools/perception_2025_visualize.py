"""Replay native artifacts in LiDAR-time ego coordinates; never substitute model outputs."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from deployment_visualize import (render, transform_points, camera_to_reference,
    box_components, track_color, bev_outline, camera_box_segments, CAMERAS, PALETTE)
from harpnext_2025 import map_labels

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('mambafusion', 'grae_frozen_reference', 'grae_mambafusion_native')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_results(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    results = data['results']
    if not isinstance(results, dict):
        raise ValueError(f'Expected token-indexed global boxes: {path}')
    return results

def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def load_artifacts(report, split):
    """Require current producer state and receipts; files alone are not evidence."""
    artifacts = {key:{} for key in FAMILIES}
    hashes = {}
    folder = report/'mambafusion'
    path = folder/f'{split}_predictions.json'
    status = read_json(folder/'status.json')
    receipt = read_json(folder/'native_receipt.json')
    if (status.get('status') in ('passed','partial') and status.get('strict_load')=='passed'
            and status.get('inference')=='passed' and status.get('official_evaluation')=='passed'
            and path.exists()):
        digest = sha(path)
        if (receipt.get('predictions',{}).get(split,{}).get('sha256')==digest
                and status.get('official_metrics',{}).get(split,{}).get('prediction_sha256')==digest):
            artifacts['mambafusion']=load_results(path);hashes['mambafusion']=digest
    for family in ('frozen_reference','mambafusion_native'):
        folder=report/'grae3dmot'/family
        status=read_json(folder/'status.json')
        if not status and family=='frozen_reference':
            legacy=read_json(report/'grae3dmot/status.json')
            if legacy.get('input_family')==family:status=legacy
        path=folder/split/'tracking.json'
        receipt=read_json(folder/'native_receipt.json')
        strict=read_json(report/'grae3dmot/strict_load.json')
        strict_ok=(status.get('strict_load')=='passed' or
            (strict.get('strict') is True and strict.get('missing')==[] and strict.get('unexpected')==[]
             and strict.get('checkpoint_sha256')==receipt.get('checkpoint_sha256') and bool(receipt)))
        if not (status.get('status') in ('passed','partial') and status.get('input_family')==family
                and status.get('inference')=='passed' and status.get('official_evaluation')=='passed'
                and strict_ok and path.exists()):continue
        receipt=read_json(folder/'native_receipt.json');digest=sha(path)
        if family=='mambafusion_native' or receipt:
            if receipt.get('tracking_sha256',{}).get(split)!=digest:continue
            manifest=folder/'input_manifest.json'
            if not manifest.exists() or sha(manifest)!=receipt.get('input_manifest_sha256'):continue
            if family=='mambafusion_native':
                source=[row for row in read_json(manifest).get('splits',[]) if row['split']==split]
                if not source or source[0].get('sha256')!=hashes.get('mambafusion'):continue
        artifacts['grae_'+family]=load_results(path);hashes['grae_'+family]=digest
    folder=report/'harpnext';harp_path=folder/'inference_frames.json'
    status=read_json(folder/'status.json');progress=read_json(folder/'inference_progress.json')
    harp={}
    native_complete=(status.get('status')=='completed_native_inference' or
        (status.get('status') in ('passed','partial') and status.get('native_evaluation')=='passed'))
    if (native_complete and status.get('strict_load',{}).get('status')=='passed'
            and status.get('inference_frames')==404 and progress.get('status')=='complete' and harp_path.exists()):
        rows=read_json(harp_path)
        if len(rows)!=status.get('inference_frames') or len(rows)!=progress.get('inference_frames'):
            raise ValueError('HARP current manifest frame count mismatch')
        for row in rows:
            token=row['sample_token']
            if token in harp:raise ValueError('Duplicate HARP sample token')
            if not row.get('prediction_sha256'):raise ValueError('Missing HARP prediction receipt')
            harp[token]=row
        hashes['harpnext_manifest']=sha(harp_path)
    return artifacts,harp,hashes

def original_labels(row, count):
    path = Path(row['prediction_path'])
    if not path.is_absolute():
        path = ROOT.parent/path
    if row.get('prediction_sha256') and sha(path) != row['prediction_sha256']:
        raise ValueError('HARP prediction checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        labels = data['prediction']
        if int(data['original_count']) != count or not np.array_equal(data['original_index'], np.arange(count)):
            raise ValueError('HARP original-point alignment mismatch')
        if labels.shape != (count,) or labels.dtype.kind not in 'ui' or np.any(labels >= 16):
            raise ValueError('HARP labels must cover every original point with classes 0..15')
        return labels.copy()

def semantic_colors(labels):
    labels = np.asarray(labels)
    colors = np.full((len(labels),3), [55,55,55], dtype=np.uint8)
    valid = labels != 255
    colors[valid] = PALETTE[labels[valid]+1]
    return colors

def ego_boxes(rows, pose, family, threshold=.25):
    inverse = np.linalg.inv(pose)
    q = Quaternion(matrix=pose[:3,:3])
    boxes = []
    for index,b in enumerate(rows):
        tracking = 'tracking_score' in b
        score = b['tracking_score' if tracking else 'detection_score']
        if score < threshold:
            continue
        name = b['tracking_name' if tracking else 'detection_name']
        ident = str(b['tracking_id']) if tracking else name
        boxes.append(dict(center_xyz=transform_points(np.asarray([b['translation']]),inverse)[0].tolist(),
            size_wlh=b['size'],rotation_wxyz=(q.inverse*Quaternion(b['rotation'])).elements.tolist(),
            score=score,class_name=name,track_id=family+':'+ident))
    return boxes

def project_points(points, camera_transform, intrinsics, width, height):
    camera = transform_points(points, np.linalg.inv(camera_transform))
    front = camera[:,2] > .1
    pixels = camera @ np.asarray(intrinsics).T
    pixels = pixels[:,:2]/np.where(front,camera[:,2],1)[:,None]
    mask = front & (pixels[:,0]>=0) & (pixels[:,0]<width) & (pixels[:,1]>=0) & (pixels[:,1]<height)
    return pixels[mask], mask

def audit_recording(path, packets, counts):
    import rerun.dataframe as rdf
    recording = rdf.load_recording(str(path))
    def table(entity):
        return recording.view(index='frame',contents=entity,include_tombstone_columns=True).select().read_all()
    def values(entity, component):
        t = table(entity)
        return dict(zip(t['frame'].to_pylist(),t['/'+entity+':'+component].to_pylist()))
    frames = list(range(len(packets)))
    lidar = values('sensors/lidar','Position3D')
    assert list(lidar) == frames
    error = 0.
    for i,p in enumerate(packets):
        entry = p['sensors']['LIDAR_TOP']
        raw = np.fromfile(entry['path'],dtype=np.float32).reshape(-1,5)[:,:3]
        expected = transform_points(raw,np.asarray(entry['T_sensor_to_ego']))
        actual = np.asarray(lidar[i])
        assert len(actual)==counts[i]['original_lidar_points'] and np.isfinite(actual).all()
        error = max(error,float(np.max(np.abs(actual-expected))))
    assert error < 1e-4
    for entity in ['models','bev/models','ground_truth','bev/ground_truth']+['cameras/'+cam for cam in CAMERAS]:
        clears = values(entity,'ClearIsRecursive')
        assert list(clears)==frames and all(x==[True] for x in clears.values())
    assert [x[0] for x in values('sample','Text').values()] == [p['sample_token'] for p in packets]
    decoded = {}
    for family in FAMILIES:
        entity='models/'+family+'/boxes'
        if not any(x['model_boxes'][family] for x in counts):
            continue
        boxes=values(entity,'PoseTranslation3D')
        for i,row in enumerate(counts):
            assert len(boxes.get(i) or []) == row['model_boxes'][family]
        decoded[family]=sum(row['model_boxes'][family] for row in counts)
    for layer in ('models/harpnext','ground_truth/semantics'):
        field='harp_points' if layer.startswith('models') else 'gt_semantic_points'
        if not any(row[field] for row in counts):
            continue
        points=values(layer,'Position3D')
        for i,row in enumerate(counts):
            actual=np.asarray(points.get(i) or [])
            assert len(actual)==row[field]
            if len(actual):
                assert np.allclose(actual,np.asarray(lidar[i]),atol=1e-4)
        decoded[layer]=sum(row[field] for row in counts)
    projected={}
    for family,field in [('harpnext','harp_points'),('ground_truth','gt_semantic_points')]:
        for cam in CAMERAS:
            entity='cameras/'+cam+'/image/'+family+'/semantics'
            if not any(row[field] and row['projected_original_points'][cam] for row in counts):continue
            points=values(entity,'Position2D')
            total=0
            for i,row in enumerate(counts):
                expected=row['projected_original_points'][cam] if row[field] else 0
                assert len(points.get(i) or [])==expected
                total+=expected
            projected[family+'/'+cam]=total
    return dict(status='passed',decoded_frames=len(frames),decoded_points=sum(x['original_lidar_points'] for x in counts),
        float32_position_max_abs_m=error,decoded_layer_counts=decoded,decoded_camera_semantic_points=projected,token_order_exact=True,box_counts_exact=True,
        representative_frames=[0,len(frames)//2,len(frames)-1],recursive_clear_frames=len(frames))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/perception_2025.yaml')
    ap.add_argument('--split',choices=['scene0061','mini_val'],default='scene0061')
    args=ap.parse_args()
    cp=Path(args.config);cp=cp if cp.is_absolute() else ROOT/cp
    config=json.loads(cp.read_text());base=ROOT/config['report']
    suffix='' if args.split=='scene0061' else '/mini_val'
    report=base/('visualization'+suffix);report.mkdir(parents=True,exist_ok=True)
    (report/'status.json').write_text(json.dumps(dict(status='running',export='running',browser_validation='pending',split=args.split)))
    out=ROOT/('outputs/rerun/deployment_4090/perception_2025'+suffix);out.mkdir(parents=True,exist_ok=True)
    packets=json.loads((ROOT/config['splits'][args.split]).read_text())
    assert len(packets)==(39 if args.split=='scene0061' else 81)
    tokens=[p['sample_token'] for p in packets];assert len(set(tokens))==len(tokens)
    artifacts,harp,hashes=load_artifacts(base,args.split)
    deim_path=ROOT/'runs/perception_2025/deimv2/mini_camera_predictions.json'
    deim={}
    if deim_path.exists():
        expected={(p['sample_token'],cam):p['sensors'][cam]['path'] for p in packets for cam in CAMERAS}
        for row in json.loads(deim_path.read_text()):
            key=(row['sample_token'],row['camera'])
            if key not in expected:continue
            assert key not in deim and Path(row['image']).resolve()==Path(expected[key]).resolve()
            deim[key]=row
    gt={}
    for row in json.loads((ROOT/config['data']/'v1.0-mini/sample_annotation.json').read_text()):
        gt.setdefault(row['sample_token'],[]).append(row)
    gt_semantics={row['token']:row['filename'] for row in json.loads((ROOT/config['data']/'v1.0-mini/lidarseg.json').read_text())}
    import rerun as rr
    import rerun.blueprint as rrb
    rr.init('CARperception-2025-'+args.split,strict=True)
    rr.save(str(out/'perception_2025.building.rrd'));rr.log('/',rr.ViewCoordinates.FLU,static=True)
    tabs=[]
    for family in (*FAMILIES,'harpnext','ground_truth'):
        layer='ground_truth' if family=='ground_truth' else 'models/'+family
        tabs.append(rrb.Horizontal(rrb.Vertical(rrb.Spatial3DView(name=family,origin='/',contents=['+ /'+layer+'/**']+([] if family in ('harpnext','ground_truth') else ['+ /sensors/lidar'])),
            rrb.Spatial2DView(name='BEV '+family,origin='bev',contents=['+ /bev/'+layer+'/**']+([] if family in ('harpnext','ground_truth') else ['+ /bev/lidar']))),
            rrb.Grid(*[rrb.Spatial2DView(name=cam,origin='cameras/'+cam+'/image',contents=['+ /cameras/'+cam+'/image',
                '+ /cameras/'+cam+'/image/'+family+'/**','+ /cameras/'+cam+'/image/deimv2']) for cam in CAMERAS],grid_columns=3),name=family))
    rr.send_blueprint(rrb.Blueprint(rrb.Vertical(rrb.Tabs(*tabs,active_tab=0),rrb.TextDocumentView(origin='guide'),row_shares=[9,1]),collapse_panels=True))
    (report/'status.json').write_text(json.dumps(dict(status='running',export='running',browser_validation='pending',split=args.split)))
    counts=[];histories={};last_scene=None;gap_resets=0
    for i,p in enumerate(packets):
        if p['scene_token']!=last_scene:histories={};last_scene=p['scene_token']
        rr.set_time('frame',sequence=i);rr.set_time('elapsed',duration=(p['timestamp_us']-packets[0]['timestamp_us'])/1e6)
        token=p['sample_token'];rr.log('sample',rr.TextDocument(token))
        lidar=p['sensors']['LIDAR_TOP'];pose=np.asarray(lidar['T_ego_to_global']);inverse=np.linalg.inv(pose)
        cloud=transform_points(np.fromfile(lidar['path'],dtype=np.float32).reshape(-1,5)[:,:3],np.asarray(lidar['T_sensor_to_ego']))
        assert np.isfinite(cloud).all()
        rr.log('sensors/lidar',rr.Points3D(cloud,colors=[110,130,145],radii=.03))
        rr.log('bev/lidar',rr.Points2D(-cloud[:,[1,0]],colors=[70,90,110],radii=.04))
        for entity in ['models','bev/models','ground_truth','bev/ground_truth']:
            rr.log(entity,rr.Clear(recursive=True))
        boxes={family:ego_boxes(data.get(token,[]),pose,family) for family,data in artifacts.items()}
        for family,items in boxes.items():
            if not items:continue
            centers,sizes,rots=box_components(items);colors=[track_color(b['track_id']) for b in items]
            rr.log('models/'+family+'/boxes',rr.Boxes3D(centers=centers,sizes=sizes,quaternions=rots,colors=colors,labels=[b['track_id'] for b in items],show_labels=False))
            rr.log('bev/models/'+family+'/boxes',rr.LineStrips2D([bev_outline(b) for b in items],colors=colors))
            if family.startswith('grae_'):
                for b in items:
                    ident=b['track_id'];old=histories.get(ident,[])
                    if old and old[-1][0] != i-1:old=[];gap_resets+=1
                    old=(old+[(i,transform_points(np.asarray([b['center_xyz']]),pose)[0])])[-12:];histories[ident]=old
                    if len(old)>1:
                        trail=transform_points(np.asarray([x[1] for x in old]),inverse)
                        rr.log('models/'+family+'/tracks/'+ident,rr.LineStrips3D([trail],colors=track_color(ident)))
                        rr.log('bev/models/'+family+'/tracks/'+ident,rr.LineStrips2D([-trail[:,[1,0]]],colors=track_color(ident)))
        semantics={}
        if token in harp:
            assert harp[token]['sample_data_token']==lidar['sample_data_token']
            semantics['harpnext']=semantic_colors(original_labels(harp[token],len(cloud)))
        if lidar['sample_data_token'] in gt_semantics:
            labels=map_labels(np.fromfile(ROOT/config['data']/gt_semantics[lidar['sample_data_token']],dtype=np.uint8))
            assert len(labels)==len(cloud)
            semantics['ground_truth']=semantic_colors(labels)
        for family,colors in semantics.items():
            layer='ground_truth/semantics' if family=='ground_truth' else 'models/harpnext'
            rr.log(layer,rr.Points3D(cloud,colors=colors,radii=.035))
            rr.log('bev/'+layer,rr.Points2D(-cloud[:,[1,0]],colors=colors,radii=.04))
        q=Quaternion(matrix=pose[:3,:3])
        gtboxes=[dict(center_xyz=transform_points(np.asarray([b['translation']]),inverse)[0],size_wlh=b['size'],rotation_wxyz=(q.inverse*Quaternion(b['rotation'])).elements) for b in gt.get(token,[])]
        if gtboxes:
            centers,sizes,rots=box_components(gtboxes)
            rr.log('ground_truth/boxes',rr.Boxes3D(centers=centers,sizes=sizes,quaternions=rots,colors=[60,230,120]))
            rr.log('bev/ground_truth/boxes',rr.LineStrips2D([bev_outline(b) for b in gtboxes],colors=[60,230,120]))
        projection_counts={}
        for cam in CAMERAS:
            e=p['sensors'][cam];tr=camera_to_reference(e,pose);path='cameras/'+cam
            rr.log(path,rr.Clear(recursive=True));rr.log(path,rr.Transform3D(translation=tr[:3,3],mat3x3=tr[:3,:3]))
            rr.log(path+'/image',rr.Pinhole(image_from_camera=e['K'],width=e['width'],height=e['height'],camera_xyz=rr.ViewCoordinates.RDF))
            rr.log(path+'/image',rr.EncodedImage(path=e['path']))
            for family,items in {**boxes,'ground_truth':gtboxes}.items():
                segments=[];colors=[]
                for b in items:
                    seg=camera_box_segments(b,tr,e['K'],e['width'],e['height']);segments+=seg
                    colors += [track_color(b['track_id']) if 'track_id' in b else [60,230,120]]*len(seg)
                if segments:rr.log(path+'/image/'+family+'/boxes',rr.LineStrips2D(segments,colors=colors))
            pixels,mask=project_points(cloud,tr,e['K'],e['width'],e['height']);projection_counts[cam]=int(mask.sum())
            for family,colors in semantics.items():
                if len(pixels):rr.log(path+'/image/'+family+'/semantics',rr.Points2D(pixels,colors=colors[mask],radii=1.))
            item=deim.get((token,cam))
            if item:
                mask=np.asarray(item['scores'])>=.25;xy=np.asarray(item['boxes_xyxy']).reshape(-1,4)[mask]
                rr.log(path+'/image/deimv2',rr.Boxes2D(mins=xy[:,:2],sizes=xy[:,2:]-xy[:,:2],colors=[255,180,70],show_labels=False))
        missing=[family for family,data in artifacts.items() if token not in data]+(['harpnext'] if token not in harp else [])
        row=dict(frame=i,sample_token=token,timestamp_us=p['timestamp_us'],original_lidar_points=len(cloud),
            model_boxes={key:len(value) for key,value in boxes.items()},harp_points=len(cloud) if 'harpnext' in semantics else 0,
            gt_semantic_points=len(cloud) if 'ground_truth' in semantics else 0,missing_models=missing,
            deim_cameras=sum((token,cam) in deim for cam in CAMERAS),projected_original_points=projection_counts)
        counts.append(row)
        rr.log('guide',rr.TextDocument('Model tabs select native MambaFusion, two GRAE input families, HARP-NeXt, or independent GT. Display threshold .25; full original point clouds. Camera calibration includes asynchronous ego poses. Missing layers cleared every frame.\n'+json.dumps(row)))
        # The lightweight overview is an explicitly named native detection snapshot; full layer selection is in Rerun.
        im=render(p,'Native MambaFusion (missing = empty)',boxes['mambafusion'],cloud,None,[],i,{})
        from PIL import ImageDraw
        draw=ImageDraw.Draw(im)
        draw.rectangle((1340,15,1919,55),fill=(8,14,22))
        draw.text((1340,22),f'frame {i:02d}/{len(packets)-1} | measured',fill=(180,205,220))
        for ci,cam in enumerate(CAMERAS):
            item=deim.get((token,cam))
            if not item:continue
            e=p['sensors'][cam];ox=640+(ci%3)*426;oy=62+(ci//3)*240
            for score,xy in zip(item['scores'],item['boxes_xyxy']):
                if score>=.25:
                    x1,y1,x2,y2=xy
                    draw.rectangle((ox+x1*426/e['width'],oy+max(25,y1*240/e['height']),ox+x2*426/e['width'],oy+max(25,y2*240/e['height'])),outline=(255,180,70),width=2)
        im.save(out/f'frame_{i:02d}.jpg',quality=86)
        if i in [0,len(packets)//2,len(packets)-1]:im.save(report/f'frame_{i:02d}.jpg',quality=90)
    rr.get_global_data_recording().flush();rr.disconnect()
    (out/'perception_2025.building.rrd').replace(out/'perception_2025.rrd')
    audit=audit_recording(out/'perception_2025.rrd',packets,counts)
    (out/'data.json').write_text(json.dumps(counts))
    url='http://127.0.0.1:9094/perception_2025'+suffix+'/'
    from urllib.parse import quote
    rerun_url='http://127.0.0.1:9095/?url='+quote(url+'perception_2025.rrd',safe='')+'&renderer=webgl&persist=0'
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font-family:sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}</style><title>Measured 2025 perception</title><h1>2025 感知实测回放</h1><p>Rerun tabs: MambaFusion / GRAE frozen / GRAE native / HARP / independent GT. Below: native MambaFusion snapshot.</p><a href="'''+rerun_url+'''">打开 Rerun 模型选择</a><input id="frame" type="range" min="0" max="'''+str(len(packets)-1)+'''" value="0"><pre id="info"></pre><img id="image"><script>fetch('data.json').then(r=>r.json()).then(d=>{function update(){image.src='frame_'+String(frame.value).padStart(2,'0')+'.jpg';info.textContent=JSON.stringify(d[+frame.value],null,2)}frame.oninput=update;update()})</script>''')
    summary=dict(status='partial',export='passed',
        browser_validation='pending',split=args.split,frames=len(packets),counts=counts,artifact_sha256=hashes,
        rrd=str(out/'perception_2025.rrd'),rrd_sha256=sha(out/'perception_2025.rrd'),url=url,rerun_url=rerun_url,
        decoded_rrd_audit=audit,trajectory_gap_resets=gap_resets,gt_separate_entity=True,
        missing_models=sorted({model for row in counts for model in row['missing_models']}),generated_at=datetime.now(timezone.utc).isoformat())
    (report/'status.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(frames=len(packets),output=str(out),audit=audit)))

if __name__=='__main__':
    main()
