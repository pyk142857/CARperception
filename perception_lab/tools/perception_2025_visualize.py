"""Export only available 2025 measurements on synchronized mini packets."""
import argparse,json,hashlib,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from deployment_visualize import render,transform_points,camera_to_reference,box_components,track_color,bev_outline,camera_box_segments,CAMERAS
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit_recording(path, packets, counts):
 import rerun.dataframe as rdf
 r=rdf.load_recording(str(path));frames=list(range(len(packets)))
 def table(entity):return r.view(index='frame',contents=entity,include_tombstone_columns=True).select().read_all()
 t=table('sensors/lidar');assert t['frame'].to_pylist()==frames
 point_error=0.
 for i,p in enumerate(packets):
  entry=p['sensors']['LIDAR_TOP'];raw=np.fromfile(entry['path'],dtype=np.float32).reshape(-1,5)[:,:3]
  expected=transform_points(raw,np.asarray(entry['T_sensor_to_ego']))
  actual=np.asarray(t['/sensors/lidar:Position3D'][i].as_py())
  assert len(actual)==counts[i]['original_lidar_points'] and np.isfinite(actual).all()
  point_error=max(point_error,float(np.max(np.abs(actual-expected))))
 assert point_error<1e-4
 clears=0
 for entity in ['models','bev/models','ground_truth']+['cameras/'+cam for cam in CAMERAS]:
  t=table(entity);assert t['frame'].to_pylist()==frames
  assert all(x==[True] for x in t['/'+entity+':ClearIsRecursive'].to_pylist());clears+=t.num_rows
 t=table('sample');assert [x[0] for x in t['/sample:Text'].to_pylist()]==[p['sample_token'] for p in packets]
 t=table('models/grae/boxes');assert [len(x) for x in t['/models/grae/boxes:PoseTranslation3D'].to_pylist()]==[x['grae_boxes'] for x in counts]
 return dict(status='passed',decoded_frames=len(frames),decoded_points=sum(x['original_lidar_points'] for x in counts),float32_position_max_abs_m=point_error,decoded_recursive_clears=clears,token_order_exact=True,box_counts_exact=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/perception_2025.yaml');a=ap.parse_args();cp=Path(a.config);cp=cp if cp.is_absolute() else ROOT/cp;c=json.loads(cp.read_text());report=ROOT/c['report']/'visualization';report.mkdir(exist_ok=True);(report/'status.json').write_text(json.dumps(dict(status='running',export='running',browser_validation='pending',generated_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n');out=ROOT/'outputs/rerun/deployment_4090/perception_2025';out.mkdir(parents=True,exist_ok=True)
 packets=json.loads((ROOT/c['splits']['scene0061']).read_text());predpath=ROOT/c['report']/'grae3dmot/frozen_reference/scene0061/tracking.json'
 if not predpath.exists():raise FileNotFoundError('No measured GRAE scene output')
 tracking=json.loads(predpath.read_text())['results'];tokens=[p['sample_token'] for p in packets];assert len(packets)==39 and len(set(tokens))==39;assert set(tracking)==set(tokens)
 deimpath=ROOT/'runs/perception_2025/deimv2/mini_camera_predictions.json';deim={}
 if deimpath.exists():
  expected={(p['sample_token'],cam):p['sensors'][cam]['path'] for p in packets for cam in CAMERAS}
  for x in json.loads(deimpath.read_text()):
   key=(x['sample_token'],x['camera']);assert key in expected and key not in deim
   assert Path(x['image']).resolve()==Path(expected[key]).resolve();deim[key]=x
 import rerun as rr
 import rerun.blueprint as rrb
 rr.init('CARperception-2025-measured',strict=True);rr.save(str(out/'perception_2025.building.rrd'));rr.log('/',rr.ViewCoordinates.FLU,static=True)
 rr.log('guide',rr.TextDocument('2025 measured outputs. GRAE uses frozen reference detections; DEIM camera boxes if available. Missing Mamba/HARP predictions are not filled from old models. Display score >= .25; evaluation remains unfiltered. GT is a separate layer. Full original point clouds.'),static=True)
 rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(rrb.Vertical(rrb.Tabs(rrb.Spatial3DView(name='2025 predictions',origin='/',contents=['+ /models/**','+ /sensors/**']),rrb.Spatial3DView(name='Independent GT',origin='/',contents=['+ /ground_truth/**','+ /sensors/**']),rrb.TextDocumentView(origin='guide'),active_tab=0),rrb.Spatial2DView(name='BEV',origin='bev')),rrb.Grid(*[rrb.Spatial2DView(name=c,origin='cameras/'+c+'/image') for c in CAMERAS],grid_columns=3)),collapse_panels=True))
 gt={}
 for x in json.loads((ROOT/c['data']/'v1.0-mini/sample_annotation.json').read_text()):gt.setdefault(x['sample_token'],[]).append(x)
 counts=[];history={};lastscene=None;coordinate_error=0.;clear_count=0;gap_resets=0
 for i,p in enumerate(packets):
  if p['scene_token']!=lastscene:history={};lastscene=p['scene_token']
  rr.set_time('frame',sequence=i);rr.set_time('elapsed',duration=(p['timestamp_us']-packets[0]['timestamp_us'])/1e6)
  missing=[cam for cam in CAMERAS if (p['sample_token'],cam) not in deim]
  rr.log('guide',rr.TextDocument('Sample '+p['sample_token']+'\nMissing DEIM cameras: '+str(missing)+'\nMambaFusion / HARP: predictions unavailable. GRAE uses frozen reference detections. Full original points. GT separate.'))
  rr.log('sample',rr.TextDocument(p['sample_token']));lidar=p['sensors']['LIDAR_TOP'];pose=np.array(lidar['T_ego_to_global']);inv=np.linalg.inv(pose);q=Quaternion(matrix=pose[:3,:3]);cloud=np.fromfile(lidar['path'],dtype=np.float32).reshape(-1,5)[:,:3];cloud=transform_points(cloud,np.array(lidar['T_sensor_to_ego']))
  assert np.isfinite(cloud).all() and len(cloud)==Path(lidar['path']).stat().st_size//20
  roundtrip=transform_points(transform_points(cloud,pose),inv);coordinate_error=max(coordinate_error,float(np.max(np.abs(roundtrip-cloud))))
  rr.log('sensors/lidar',rr.Points3D(cloud,colors=[110,130,145],radii=.03));rr.log('bev/lidar',rr.Points2D(-cloud[:,[1,0]],colors=[70,90,110],radii=.04))
  rr.log('models',rr.Clear(recursive=True));rr.log('bev/models',rr.Clear(recursive=True));rr.log('ground_truth',rr.Clear(recursive=True));clear_count+=3;boxes=[]
  for b in tracking[p['sample_token']]:
   if b['tracking_score']<.25:continue
   boxes.append(dict(center_xyz=transform_points(np.array([b['translation']]),inv)[0].tolist(),size_wlh=b['size'],rotation_wxyz=(q.inverse*Quaternion(b['rotation'])).elements.tolist(),score=b['tracking_score'],class_name=b['tracking_name'],track_id=b['tracking_id']))
  centers,sizes,rots=box_components(boxes);colors=[track_color(b['track_id']) for b in boxes]
  rr.log('models/grae/boxes',rr.Boxes3D(centers=centers,sizes=sizes,quaternions=rots,colors=colors,labels=[b['track_id'].split(':')[-1] for b in boxes],show_labels=False));rr.log('bev/models/grae/boxes',rr.LineStrips2D([bev_outline(b) for b in boxes],colors=colors))
  trails={}
  for b in boxes:
   ident=b['track_id'];old=history.get(ident,[])
   if old and i-old[-1][0]>1:old=[];gap_resets+=1
   old.append((i,transform_points(np.array([b['center_xyz']]),pose)[0]));old=old[-12:];history[ident]=old;trails[ident]=transform_points(np.array([x[1] for x in old]),inv)
   if len(old)>1:rr.log('models/grae/tracks/'+ident,rr.LineStrips3D([trails[ident]],colors=track_color(ident)))
  gtboxes=[dict(center_xyz=transform_points(np.array([x['translation']]),inv)[0].tolist(),size_wlh=x['size'],rotation_wxyz=(q.inverse*Quaternion(x['rotation'])).elements.tolist()) for x in gt.get(p['sample_token'],[])];gc,gs,gq=box_components(gtboxes);rr.log('ground_truth/boxes',rr.Boxes3D(centers=gc,sizes=gs,quaternions=gq,colors=[60,230,120]))
  for cam in CAMERAS:
   e=p['sensors'][cam];tr=camera_to_reference(e,pose);path='cameras/'+cam;rr.log(path,rr.Clear(recursive=True));clear_count+=1;rr.log(path,rr.Transform3D(translation=tr[:3,3],mat3x3=tr[:3,:3]));rr.log(path+'/image',rr.Pinhole(image_from_camera=e['K'],width=e['width'],height=e['height'],camera_xyz=rr.ViewCoordinates.RDF));rr.log(path+'/image',rr.EncodedImage(path=e['path']));segments=[];cols=[]
   for b in boxes:
    seg=camera_box_segments(b,tr,e['K'],e['width'],e['height']);segments+=seg;cols += [track_color(b['track_id'])]*len(seg)
   if segments:rr.log(path+'/image/grae',rr.LineStrips2D(segments,colors=cols))
   item=deim.get((p['sample_token'],cam))
   if item:
    mask=np.array(item['scores'])>=.25;xy=np.array(item['boxes_xyxy']).reshape(-1,4)[mask];rr.log(path+'/image/deimv2',rr.Boxes2D(mins=xy[:,:2],sizes=xy[:,2:]-xy[:,:2],colors=[255,180,70],show_labels=False))
  im=render(p,'GRAE frozen input + DEIMv2-S cameras',boxes,cloud,None,[],i,trails)
  from PIL import ImageDraw
  draw=ImageDraw.Draw(im)
  for ci,cam in enumerate(CAMERAS):
   item=deim.get((p['sample_token'],cam))
   if item:
    e=p['sensors'][cam];ox=640+(ci%3)*426;oy=62+(ci//3)*240
    for score,xy in zip(item['scores'],item['boxes_xyxy']):
     if score>=.25:
      x1,y1,x2,y2=xy;draw.rectangle((ox+x1*426/e['width'],oy+max(25,y1*240/e['height']),ox+x2*426/e['width'],oy+max(25,y2*240/e['height'])),outline=(255,180,70),width=2)
  im.save(out/f'frame_{i:02d}.jpg',quality=86)
  if i in [0,19,38]:im.save(report/f'frame_{i:02d}.jpg',quality=90)
  counts.append(dict(frame=i,sample_token=p['sample_token'],timestamp_us=p['timestamp_us'],original_lidar_points=len(cloud),grae_boxes=len(boxes),deim_cameras=sum((p['sample_token'],cam) in deim for cam in CAMERAS)))
 rr.get_global_data_recording().flush();rr.disconnect();(out/'perception_2025.building.rrd').replace(out/'perception_2025.rrd')
 (out/'data.json').write_text(json.dumps(counts));(out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font-family:sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}</style><title>2025 measured perception</title><h1>2025 感知回放</h1><p>GRAE + 封存检测，DEIMv2 2D 结果在 Rerun 中（资源齐全时）。缺失模型不会沿用旧帧。</p><a href="http://127.0.0.1:9095/?url=http%3A%2F%2F127.0.0.1%3A9094%2Fperception_2025%2Fperception_2025.rrd&renderer=webgl">打开 Rerun</a><input id="frame" type="range" min="0" max="38" value="0"><pre id="info"></pre><img id="image" style="width:100%"><script>fetch('data.json').then(r=>r.json()).then(d=>{function update(){image.src='frame_'+String(frame.value).padStart(2,'0')+'.jpg';info.textContent=JSON.stringify(d[+frame.value],null,2)}frame.oninput=update;update()})</script>''')
 decoded_audit=audit_recording(out/'perception_2025.rrd',packets,counts)
 summary=dict(decoded_rrd_audit=decoded_audit,status='partial',export='passed',browser_validation='pending',frames=len(packets),counts=counts,tracking_sha256=sha(predpath),deim_sha256=sha(deimpath) if deimpath.exists() else None,rrd_sha256=sha(out/'perception_2025.rrd'),rrd=str(out/'perception_2025.rrd'),url='http://127.0.0.1:9094/perception_2025/',missing_models=['mambafusion','harpnext'],trajectory_gap_clear=True,validation=dict(original_points_total=sum(x['original_lidar_points'] for x in counts),coordinate_roundtrip_max_abs_m=coordinate_error,recursive_clears=clear_count,expected_recursive_clears=39*9,trajectory_gap_resets=gap_resets,gt_separate_entity=True,deim_camera_pairs=len(deim),finite_points=True),generated_at=datetime.now(timezone.utc).isoformat())
 (report/'status.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'frames':len(packets),'output':str(out)}))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  config=Path(sys.argv[sys.argv.index('--config')+1]) if '--config' in sys.argv else ROOT/'configs/perception_2025.yaml'
  if not config.is_absolute():config=ROOT/config
  report=ROOT/json.loads(config.read_text())['report']/'visualization';report.mkdir(parents=True,exist_ok=True)
  (report/'status.json').write_text(json.dumps(dict(status='failed',export='failed',browser_validation='not_run',error=str(exc),traceback=traceback.format_exc(),generated_at=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
  raise
