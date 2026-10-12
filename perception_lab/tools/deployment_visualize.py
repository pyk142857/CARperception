"""Synchronized measured deployment Rerun + videos + local frame inspection page."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pyquaternion import Quaternion
from tracking_overlay import box_corners,bev_outline,camera_box_segments,track_color,EDGES
from rerun_mini import camera_to_reference,box_components,transform_points,CAMERAS

LAB=Path(__file__).resolve().parents[1]
REPORT=LAB/'reports/deployment_4090_20261008/visualize'
OUT=LAB/'outputs/rerun/deployment_4090'
PALETTE=np.array([[0,0,0],[255,120,60],[255,70,150],[255,200,40],[80,160,255],[190,100,40],[190,90,240],[255,70,70],[255,160,0],[210,130,170],[130,130,240],[30,190,190],[120,210,210],[220,160,210],[155,150,75],[160,170,180],[90,180,80],[0,0,0]],dtype=np.uint8)
MODEL_COLORS={'bevfusion_reference':(255,170,60),'bevfusion_tensorrt':(70,225,210),'sparse4d':(130,160,255)}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def project_cloud(xyz,kind):
    if kind=='bev':return np.c_[320-xyz[:,1]*6,580-xyz[:,0]*6]
    # Fixed orthographic virtual view; display axes shown in the legend.
    return np.c_[640+(xyz[:,0]-xyz[:,1])*6,310-(xyz[:,0]+xyz[:,1])*2.7-xyz[:,2]*12]

def dot_raster(xyz,colors,size,kind,radius=0):
    arr=np.full((size[1],size[0],3),[13,22,31],np.uint8)
    xy=np.rint(project_cloud(xyz,kind)).astype(int)
    mask=(xy[:,0]>=0)&(xy[:,0]<size[0])&(xy[:,1]>=0)&(xy[:,1]<size[1])
    xy=xy[mask];selected=colors[mask] if np.ndim(colors)==2 else colors
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            arr[np.clip(xy[:,1]+dy,0,size[1]-1),np.clip(xy[:,0]+dx,0,size[0]-1)]=selected
    return Image.fromarray(arr)

def camera_tile(packet,cam,boxes,model):
    entry=packet['sensors'][cam];im=Image.open(entry['path']).convert('RGB');draw=ImageDraw.Draw(im)
    pose=np.array(packet['sensors']['LIDAR_TOP']['T_ego_to_global']);trans=camera_to_reference(entry,pose)
    for b in boxes:
        col=track_color(b['track_id']) if 'track_id' in b else MODEL_COLORS.get(model,(255,170,60))
        segments=camera_box_segments(b,trans,entry['K'],entry['width'],entry['height'])
        for seg in segments:draw.line([tuple(x) for x in seg],fill=col,width=4)
        if segments and 'native_instance_id' in b:
            xy=np.asarray(segments).reshape(-1,2);anchor=xy[np.argmin(xy[:,1])]
            draw.text(tuple(anchor),str(b['native_instance_id']),font=font(23),fill=col,stroke_width=1)
    im=im.resize((426,240));draw=ImageDraw.Draw(im);draw.rectangle((0,0,425,24),fill=(10,18,28));draw.text((8,3),cam,font=font(15),fill=(230,235,245))
    return im

def render(packet,model,boxes,cloud,occ,classes,idx,trails):
    im=Image.new('RGB',(1920,1080),(8,14,22));draw=ImageDraw.Draw(im)
    title={'sparse4d':'Sparse4D v3 / native temporal IDs','gaussianworld':'GaussianWorld / semantic occupancy','bevfusion_reference':'BEVFusion R50 / PyTorch reference','bevfusion_tensorrt':'BEVFusion R50 / TensorRT FP16'}.get(model,model)
    draw.text((22,15),title,font=font(28),fill=(240,245,255));draw.text((1340,22),f'frame {idx:02d}/38 | measured | 2 fps playback',font=font(19),fill=(180,205,220))
    if occ is not None:xyz,labels,slice_mask=occ;cols=PALETTE[labels]
    else:xyz=cloud;cols=np.tile([95,115,125],(len(xyz),1))
    bev=dot_raster(xyz[slice_mask] if occ is not None else xyz,cols[slice_mask] if occ is not None else cols,(640,950),'bev',radius=1 if occ is not None else 0);bd=ImageDraw.Draw(bev)
    for tick in range(-40,51,10):
        y=580-tick*6;bd.line([(0,y),(640,y)],fill=(34,45,55));bd.text((4,y+2),f'{tick}m',font=font(12),fill=(105,130,145))
    bd.line([(320,0),(320,950)],fill=(55,70,80))
    for b in boxes:
        col=track_color(b['track_id']) if 'track_id' in b else MODEL_COLORS.get(model,(255,170,60));pts=project_cloud(box_corners(b)[[0,1,2,3,0]],'bev');bd.line([tuple(x) for x in pts],fill=col,width=2)
        if 'native_instance_id' in b:
            p=project_cloud(np.array([b['center_xyz']]),'bev')[0];bd.text(tuple(p),str(b['native_instance_id']),font=font(12),fill=col)
    for ident,points in trails.items():
        if len(points)>1:bd.line([tuple(x) for x in project_cloud(points,'bev')],fill=track_color(ident),width=2)
    bd.polygon([(320,563),(313,582),(327,582)],fill=(245,245,245));bd.text((30,905),'LIDAR z=-1.75m slice -> ego | forward up' if occ is not None else 'Forward = up | left = left | ego metres',font=font(15),fill=(210,225,235))
    im.paste(bev,(0,62))
    for j,cam in enumerate(CAMERAS):im.paste(camera_tile(packet,cam,boxes,model),(640+(j%3)*426,62+(j//3)*240))
    oblique=dot_raster(xyz,cols,(1280,480),'oblique',radius=1 if occ is not None else 0);od=ImageDraw.Draw(oblique)
    for b in boxes:
        col=track_color(b['track_id']) if 'track_id' in b else MODEL_COLORS.get(model,(255,170,60));corners=project_cloud(box_corners(b),'oblique')
        for edge in EDGES:od.line([tuple(corners[edge[0]]),tuple(corners[edge[1]])],fill=col,width=2)
    od.text((18,12),'Ego 3D orthographic view | x forward, y left, z up',font=font(17),fill=(210,225,235))
    im.paste(oblique,(640,548));draw=ImageDraw.Draw(im)
    if occ is not None:
        for cl in range(1,17):
            x=24+((cl-1)%8)*235;y=1017+((cl-1)//8)*20
            draw.rectangle((x,y,x+12,y+12),fill=tuple(PALETTE[cl]));draw.text((x+16,y-3),classes[cl],font=font(12),fill=(215,225,235))
    else:draw.text((22,1023),f'Display threshold >= 0.25 | {len(boxes)} displayed boxes | full predictions retained for evaluation',font=font(17),fill=(180,205,220))
    draw.text((640,1059),packet['sample_token']+f"  t={packet['timestamp_us']} us",font=font(16),fill=(165,190,210))
    return im

def serve():
    OUT.mkdir(parents=True,exist_ok=True)
    state=OUT/'server.json'
    if state.exists():
        previous=json.loads(state.read_text())
        try:
            for pid in previous['pids']:os.kill(pid,0)
            print(previous['url']);return
        except OSError:pass
    for port in [9094,9095,9880]:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    binary=next((LAB/'envs/rerun/lib').glob('python*/site-packages/rerun_sdk/rerun_cli/rerun'))
    commands=[[sys.executable,str(Path(__file__).resolve()),'--http-only'],[str(binary),'--serve-web','--bind','127.0.0.1','--web-viewer-port','9095','--port','9880','--server-memory-limit','512MB','--threads','2']]
    processes=[]
    with (OUT/'server.log').open('a') as log:
        for cmd in commands:processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True))
    state.write_text(json.dumps(dict(pids=[p.pid for p in processes],commands=commands,url='http://127.0.0.1:9094/',rerun_url='http://127.0.0.1:9095/?url=http%3A%2F%2F127.0.0.1%3A9094%2Fdeployment_4090.rrd&renderer=webgl'),indent=2))
    print('http://127.0.0.1:9094/')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prediction',action='append',default=[]);p.add_argument('--gaussian-manifest',type=Path,default=LAB/'runs/deployment_4090_20261008/gaussianworld/manifest.json');p.add_argument('--packets',type=Path,default=LAB/'reports/deployment_4090_20261008/inputs/scene0061_packets.json');p.add_argument('--serve',action='store_true');p.add_argument('--http-only',action='store_true');p.add_argument('--skip-videos',action='store_true');a=p.parse_args()
    if a.http_only:
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self,*args,**kw):super().__init__(*args,directory=str(OUT),**kw)
            def end_headers(self):self.send_header('Access-Control-Allow-Origin','*');super().end_headers()
        ThreadingHTTPServer(('127.0.0.1',9094),Handler).serve_forever();return
    if a.serve and not a.prediction:serve();return
    import rerun as rr
    import rerun.blueprint as rrb
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    packets=json.loads(a.packets.read_text());tokens=[r['sample_token'] for r in packets]
    models={};sources={}
    for spec in a.prediction:
        name,path=spec.split('=',1);path=Path(path);records=json.loads(path.read_text());by_token={r['sample_token']:r for r in records}
        if set(by_token)!=set(tokens) or len(records)!=len(tokens):raise ValueError(f'Exact unique coverage required for {name}')
        if any(r.get('source')!='measured' for r in records):raise ValueError('Only measured outputs accepted')
        models[name]=[by_token[t] for t in tokens];sources[name]=dict(path=str(path.resolve()),sha256=sha(path))
    manifest=json.loads(a.gaussian_manifest.read_text());occ_by={r['sample_token']:r for r in manifest['frames']}
    if set(occ_by)!=set(tokens):raise ValueError('Occupancy coverage mismatch')
    models['gaussianworld']=[occ_by[t] for t in tokens];sources['gaussianworld']=dict(path=str(a.gaussian_manifest),sha256=sha(a.gaussian_manifest))
    rr.init('CARperception-4090-deployment',strict=True);rr.save(str(OUT/'deployment_4090.building.rrd'))
    rr.log('/',rr.ViewCoordinates.FLU,static=True)
    rr.log('models/gaussianworld',rr.AnnotationContext([rr.AnnotationInfo(id=i,label=n,color=PALETTE[i]) for i,n in enumerate(manifest['classes'])]),static=True)
    rr.log('guide',rr.TextDocument('# Measured 4090 deployment\n39 synchronized scene-0061 keyframes. Cameras retain asynchronous poses. Ego at LIDAR timestamp, x forward y left z up.\n\nModel entities are independent. Render threshold .25 does not affect official evaluations. Sparse4D labels are native IDs with scene namespaces. Trails are transformed from historical global coordinates.\n\nGaussianWorld XYZ voxel centers are converted LIDAR→ego; 0 ignore and17 empty hidden. Occupancy is predicted, no voxel GT available. Full current LiDAR shown for context, not input to camera-only models. Box GT has a separate tab/entity and never enters prediction or model videos. Occupancy slice tab is LIDAR z=-1.75m before ego transform.\n\n2fps videos are playback speed, not inference throughput.',media_type='text/markdown'),static=True)
    camera_views=[rrb.Spatial2DView(name=c,origin=f'cameras/{c}/image') for c in CAMERAS]
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(rrb.Vertical(rrb.Tabs(rrb.Spatial3DView(name='3D / independent measured models',origin='/',contents=['+ /models/**','+ /sensors/**','+ /cameras/**']),rrb.Spatial3DView(name='Ground truth / separate layer',origin='/',contents=['+ /ground_truth/**','+ /sensors/**']),rrb.TextDocumentView(origin='guide'),active_tab=0),rrb.Tabs(rrb.Spatial2DView(name='BEV / forward up',origin='bev',visual_bounds=rrb.VisualBounds2D(x_range=[-45,45],y_range=[-55,45])),rrb.Spatial2DView(name='Occupancy slice / LIDAR z=-1.75m',origin='occupancy_slice',visual_bounds=rrb.VisualBounds2D(x_range=[-45,45],y_range=[-55,45])),active_tab=0),row_shares=[.58,.42]),rrb.Grid(*camera_views,grid_columns=3),column_shares=[.44,.56]),rrb.TimePanel(state='collapsed'),collapse_panels=True))
    gt_by_token={}
    for ann in json.loads((LAB/'data/nuscenes/v1.0-mini/sample_annotation.json').read_text()):
        if ann['sample_token'] in tokens:gt_by_token.setdefault(ann['sample_token'],[]).append(ann)
    histories={name:{} for name in models};video_procs={};counts=[];start=time.perf_counter()
    for name in models:
        (OUT/name).mkdir(exist_ok=True);(REPORT/name).mkdir(exist_ok=True)
        if not a.skip_videos:
            cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x1080','-framerate','2','-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p',str(OUT/name/'scene0061.mp4')]
            video_procs[name]=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    last_scene=None
    for idx,packet in enumerate(packets):
        if packet['scene_token'] != last_scene:
            histories={name:{} for name in models};last_scene=packet['scene_token']
        rr.set_time('frame',sequence=idx);rr.set_time('elapsed',duration=(packet['timestamp_us']-packets[0]['timestamp_us'])/1e6)
        rr.log('frame_info',rr.AnyValues(sample_token=packet['sample_token'],timestamp_us=packet['timestamp_us']))
        lidar=packet['sensors']['LIDAR_TOP'];pose=np.array(lidar['T_ego_to_global']);inverse=np.linalg.inv(pose)
        cloud=np.fromfile(lidar['path'],dtype=np.float32).reshape(-1,5)[:,:3];cloud=transform_points(cloud,np.array(lidar['T_sensor_to_ego']))
        rr.log('sensors/current_lidar',rr.Points3D(cloud,colors=[110,130,145],radii=.03));rr.log('bev/lidar',rr.Points2D(-cloud[:,[1,0]],colors=[70,90,110],radii=.04))
        gt=[];q_ego=Quaternion(matrix=pose[:3,:3])
        for ann in gt_by_token.get(packet['sample_token'],[]):
            gt.append(dict(center_xyz=transform_points(np.array([ann['translation']]),inverse)[0].tolist(),size_wlh=ann['size'],rotation_wxyz=(q_ego.inverse*Quaternion(ann['rotation'])).elements.tolist()))
        gc,gs,gq=box_components(gt);rr.log('ground_truth/boxes',rr.Boxes3D(centers=gc,sizes=gs,quaternions=gq,colors=[60,230,120]))
        for cam in CAMERAS:
            entry=packet['sensors'][cam];trans=camera_to_reference(entry,pose);path='cameras/'+cam
            rr.log(path,rr.Transform3D(translation=trans[:3,3],mat3x3=trans[:3,:3]));rr.log(path+'/image',rr.Pinhole(image_from_camera=entry['K'],width=entry['width'],height=entry['height'],camera_xyz=rr.ViewCoordinates.RDF,image_plane_distance=1.5));rr.log(path+'/image',rr.EncodedImage(path=entry['path']))
        for name,records in models.items():
            rr.log('models/'+name,rr.Clear(recursive=True));rr.log('bev/models/'+name,rr.Clear(recursive=True))
            boxes=[b for b in records[idx].get('boxes3d',[]) if b['score']>=.25];occ=None;trails={}
            if name=='gaussianworld':
                rr.log('models/gaussianworld',rr.AnnotationContext([rr.AnnotationInfo(id=i,label=n,color=PALETTE[i]) for i,n in enumerate(manifest['classes'])]))
                row=records[idx];path=Path(row['occupancy_path'])
                if sha(path)!=row['sha256']:raise ValueError('Occupancy hash changed')
                data=np.load(path);grid=data['occupancy'];indices=np.argwhere((grid!=manifest['empty_index'])&(grid!=manifest['ignore_index']));labels=grid[tuple(indices.T)];xyz=indices*np.array(manifest['voxel_size_m'])+np.array(manifest['pc_range'][:3])+np.array(manifest['voxel_size_m'])/2;xyz=transform_points(xyz,data['lidar2ego']);slice_mask=indices[:,2]==6;occ=(xyz,labels,slice_mask)
                rr.log('models/'+name+'/occupancy',rr.Points3D(xyz,colors=PALETTE[labels],class_ids=labels,radii=.22));rr.log('bev/models/'+name+'/occupancy',rr.Points2D(-xyz[:,[1,0]],colors=PALETTE[labels],radii=.22));rr.log('occupancy_slice',rr.Points2D(-xyz[slice_mask][:,[1,0]],colors=PALETTE[labels[slice_mask]],radii=.22))
            else:
                centers,sizes,rotations=box_components(boxes);colors=[track_color(b['track_id']) if 'track_id' in b else MODEL_COLORS.get(name,(255,170,60)) for b in boxes];labels=[str(b.get('native_instance_id',b['class_name'])) for b in boxes]
                rr.log('models/'+name+'/boxes',rr.Boxes3D(centers=centers,sizes=sizes,quaternions=rotations,colors=colors,labels=labels,show_labels=name=='sparse4d'))
                rr.log('bev/models/'+name+'/boxes',rr.LineStrips2D([bev_outline(b) for b in boxes],colors=colors,radii=rr.Radius.ui_points(1.5)))
                for b in boxes:
                    if 'track_id' not in b:continue
                    ident=b['track_id'];history=histories[name].setdefault(ident,[]);history.append(transform_points(np.array([b['center_xyz']]),pose)[0]);history[:]=history[-12:];trails[ident]=transform_points(np.array(history),inverse)
                    if len(history)>1:rr.log('models/'+name+'/tracks/'+ident,rr.LineStrips3D([trails[ident]],colors=track_color(ident),radii=.07));rr.log('bev/models/'+name+'/tracks/'+ident,rr.LineStrips2D([-trails[ident][:,[1,0]]],colors=track_color(ident),radii=.07))
            for cam in CAMERAS:
                entry=packet['sensors'][cam];trans=camera_to_reference(entry,pose);path=f'cameras/{cam}/image/models/{name}';rr.log(path,rr.Clear(recursive=True));segments=[];colors=[]
                for b in boxes:
                    seg=camera_box_segments(b,trans,entry['K'],entry['width'],entry['height']);segments+=seg;colors += [track_color(b['track_id']) if 'track_id' in b else MODEL_COLORS.get(name,(255,170,60))]*len(seg)
                if segments:rr.log(path,rr.LineStrips2D(segments,colors=colors,radii=rr.Radius.ui_points(1)))
            frame=render(packet,name,boxes,cloud,occ,manifest['classes'],idx,trails);frame.save(OUT/name/f'frame_{idx:02d}.jpg',quality=86)
            if idx in [0,19,38]:frame.save(REPORT/name/f'frame_{idx:02d}.jpg',quality=90)
            if name in video_procs:video_procs[name].stdin.write(np.asarray(frame).tobytes())
            counts.append(dict(model=name,frame=idx,sample_token=packet['sample_token'],display_boxes=len(boxes),occupancy_voxels=len(occ[0]) if occ else None,lidar_points=len(cloud)))
        print(f'visualized {idx+1}/{len(packets)}: '+','.join(models),flush=True)
    verification={}
    for name,proc in video_procs.items():
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg encode failed')
        file=OUT/name/'scene0061.mp4';probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(file)]));subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],check=True);assert probe['streams'][0]['nb_read_frames']==str(len(packets));verification[name]=dict(path=str(file),sha256=sha(file),decode_verified=True,probe=probe)
    page_data=dict(models=list(models),frames=[dict(sample_token=p['sample_token'],timestamp_us=p['timestamp_us']) for p in packets])
    (OUT/'data.json').write_text(json.dumps(page_data))
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>CARperception · 4090 measured deployment</title><link rel="icon" href="data:,"><style>body{background:#08101a;color:#e9f0f7;font:16px system-ui;margin:22px}header{display:flex;align-items:center;gap:20px}button,select{font:inherit;padding:8px;background:#22384c;color:white;border:1px solid #53728a;border-radius:5px}a{color:#69c8ee}img{display:block;width:100%;max-height:78vh;object-fit:contain;margin-top:12px}input{width:45%}#token{font:13px monospace;color:#9db8ca}small{color:#b0c5d4}</style><header><h2>4090 · 实测感知回放</h2><select id="model"></select><a href="http://127.0.0.1:9095/?url=http%3A%2F%2F127.0.0.1%3A9094%2Fdeployment_4090.rrd&renderer=webgl" target="_blank">打开交互式 Rerun 3D</a><a id="video" target="_blank">MP4</a></header><p><button data-frame="0">首帧</button> <button data-frame="19">中帧</button> <button data-frame="38">末帧</button> <input id="frame" type="range" min="0" max="38" value="0"><b id="index"></b></p><div id="token"></div><img id="view" alt="Measured synchronized sensor and model output"><small>39 个真实关键帧 · 六相机 / BEV / 3D · 显示分数 ≥ 0.25；评估保留完整预测 · 2 fps 是视频播放速度 · 占用无真值</small><script>fetch('data.json').then(r=>r.json()).then(d=>{window.data=d;for(const m of d.models){let o=document.createElement('option');o.value=o.textContent=m;model.append(o)}function update(){let i=+frame.value,m=model.value;view.src=m+'/frame_'+String(i).padStart(2,'0')+'.jpg';index.textContent=i+' / 38';token.textContent=d.frames[i].sample_token+' | '+d.frames[i].timestamp_us+' us';video.href=m+'/scene0061.mp4'}frame.oninput=update;model.onchange=update;document.querySelectorAll('[data-frame]').forEach(b=>b.onclick=()=>{frame.value=b.dataset.frame;update()});update()})</script>''')
    rr.disconnect()
    (OUT/'deployment_4090.building.rrd').replace(OUT/'deployment_4090.rrd')
    summary=dict(status='passed',scope='RRD export and complete video decode; browser validation recorded separately',input_packet_sha256=sha(a.packets),frames=len(packets),models=list(models),sources=sources,render_threshold=.25,ground_truth='separate Rerun tab/entity from official annotations, excluded from all model videos and predictions',playback_fps=2,occupancy_bev_slice=dict(axis='LIDAR z',index=6,center_m=-1.75,thickness_m=.5,output_frame='ego'),seconds=time.perf_counter()-start,video_verification=verification,counts=counts,rrd_path=str(OUT/'deployment_4090.rrd'),rrd_sha256=sha(OUT/'deployment_4090.rrd'))
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if a.serve:serve()

if __name__=='__main__':main()
