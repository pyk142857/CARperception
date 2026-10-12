"""CPU-only audit of existing P4 evidence; writes only alongside this script."""
import csv,gzip,hashlib,json,os,pathlib,subprocess,datetime
import numpy as np
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[4]; R=ROOT/'perception_lab/reports/deployment_4090_20261008'; OUT=pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)
def j(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def artifact(p):
 p=pathlib.Path(p);return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'exists':p.exists(),**({'bytes':p.stat().st_size,'sha256':sha(p)} if p.is_file() else {})}
A={'audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':'CPU-only read-only audit of old evidence; no inference/evaluation/browser rerun','source_commit':subprocess.check_output(['git','rev-parse','ce24482'],text=True).strip(),'sparse4d':{},'gaussianworld':{},'visualization':{}}
for split in ['scene0061','mini_val']:
 d=R/'sparse4d'/split;s=j(d/'summary.json');expected=j(R/'inputs'/f'{split}_samples.json');tokens=[x['sample_token'] for x in expected]
 blobs={k:gzip.decompress((d/f'{k}.json.gz').read_bytes()) for k in ['predictions','native_predictions']};p=json.loads(blobs['predictions']);n=json.loads(blobs['native_predictions']);metrics=j(d/'detection_eval/metrics_summary.json');trk=j(d/'tracking_eval/metrics_summary.json');timings=list(csv.DictReader((d/'timings.csv').open()))
 input_order_tokens=tokens[:]; expected=sorted(expected,key=lambda x:(x['scene_token'],x['timestamp_us']));tokens=[x['sample_token'] for x in expected]
 resets=[{'index':i,'scene_token':x['scene_token'],'cache_empty':True,'prev_id':0} for i,x in enumerate(expected) if i==0 or expected[i-1]['scene_token']!=x['scene_token']]
 A['sparse4d'][split]={'frames':len(p),'expected_frames':len(tokens),'input_packet_order_identical':input_order_tokens==tokens,'ordering_contract':'sorted by (scene_token,timestamp), source sparse4d_4090.py:99','token_set_coverage':set(input_order_tokens)==set(s['sample_tokens']),'ordered_token_coverage':tokens==[x['sample_token'] for x in p]==[x['sample_token'] for x in n]==s['sample_tokens']==[x['sample_token'] for x in timings],'prediction_uncompressed_sha256_matches':hashlib.sha256(blobs['predictions']).hexdigest()==s['prediction_sha256'],'native_uncompressed_sha256_matches':hashlib.sha256(blobs['native_predictions']).hexdigest()==s['native_prediction_sha256'],'all_300_boxes':all(len(x['boxes3d'])==300 for x in p),'native_ID_preserved':all([b['native_instance_id'] for b in x['boxes3d']]==y['instance_ids'] for x,y in zip(p,n)),'scene_namespaced_IDs':all(b['track_id']==f"{x['scene_token']}:{b['native_instance_id']}" for x in p for b in x['boxes3d']),'scene_reset_evidence_matches_boundaries':s['scene_resets']==resets,'scene_resets':s['scene_resets'],'metrics':{'mAP':metrics['mean_ap'],'NDS':metrics['nd_score'],'AMOTA':trk['amota'],'AMOTP':trk['amotp'],'IDS':trk['ids']},'state_tests_existing':j(d/'state_tests.json'),'timing_existing':s['timing'],'source_files':[artifact(d/f) for f in ['summary.json','state_tests.json','detection_eval/metrics_summary.json','tracking_eval/metrics_summary.json','predictions.json.gz','native_predictions.json.gz','timings.csv']]}
g=R/'gaussianworld';gs=j(g/'summary.json');gm=j(g/'grids_manifest.json')['frames'];samples=j(R/'inputs/scene0061_samples.json');packets=j(R/'inputs/scene0061_packets.json');grids=[]
for i,f in enumerate(gm):
 path=g/f['published_grid'];z=np.load(path);a=z['occupancy'];grids.append({'sample_token':f['sample_token'],'timestamp_matches':f['timestamp_us']==samples[i]['timestamp_us'],'history_matches':f['history_source']==(None if i==0 else gm[i-1]['sample_token']),'sha256_matches':sha(path)==f['sha256'],'shape':list(a.shape),'dtype':str(a.dtype),'class_range':[int(a.min()),int(a.max())],'class_counts_match':np.bincount(a.ravel(),minlength=18).tolist()==f['class_counts'],'transforms_finite_4x4':all(z[k].shape==(4,4) and bool(np.isfinite(z[k]).all()) for k in ['lidar2ego','lidar2global']),**artifact(path)})
prov=j(g/'provenance.json');pa=[]
for f in prov['artifacts']:
 a=artifact(ROOT/f['path']);a['expected_sha256']=f['sha256'];a['matches']=a.get('sha256')==f['sha256'];pa.append(a)
# Author loader requires exact LiDAR basename + .npy, not sample-token guesses.
search_roots=[ROOT/'perception_lab/data',ROOT/'perception_lab/third_party/GaussianWorld/data',pathlib.Path('/media/minglei'),pathlib.Path('/mnt')]
expected_names={pathlib.Path(x['sensors']['LIDAR_TOP']['path']).name+'.npy':x['sample_token'] for x in packets};matches=[];candidate_dirs=[];errors=[];seen=set()
for root in search_roots:
 if not root.exists():continue
 for here,dirs,files in os.walk(root,followlinks=True,onerror=lambda e:errors.append(str(e))):
  real=os.path.realpath(here)
  if real in seen:dirs[:]=[];continue
  seen.add(real)
  if 'surroundocc' in here.lower():candidate_dirs.append(here)
  for fn in files:
   if fn in expected_names:matches.append({'sample_token':expected_names[fn],**artifact(pathlib.Path(here)/fn)})
A['gaussianworld']={'frames':len(grids),'ordered_token_coverage':[f['sample_token'] for f in gm]==[s['sample_token'] for s in samples]==gs['sample_tokens'],'gaussians':gs['gaussians'],'grid_shape':gs['grid_shape'],'axis_order':gs['axis_order'],'coordinate_frame':gs['coordinate_frame'],'protocol':gs['protocol'],'inference_status':gs['inference_status'],'evaluation_status':'blocked_data' if not matches else 'GT_candidates_require_provenance_validation','label_independence_existing':j(g/'audit.json'),'grids':grids,'provenance_checks':pa,'gt_search':{'roots':[str(x) for x in search_roots],'directories_visited':len(seen),'candidate_directories':candidate_dirs,'errors':errors,'exact_LIDAR_filename_matches':matches,'token_to_required_label_file':[{'sample_token':t,'filename':f} for f,t in expected_names.items()],'scope_limit':'Local configured data/author data and mounted storage only; no downloads, no remote catalogue query; LiDARSeg not accepted as occupancy GT.'}}
v=j(R/'visualize/summary.json');b=j(R/'visualize/browser_qa.json');rr=j(R/'visualize/rerun_browser_qa.json');imgs=[]
for pattern in ['browser_*_00.png','browser_*_19.png','browser_*_38.png','rerun_browser_*.png','*/frame_00.jpg','*/frame_19.jpg','*/frame_38.jpg']:
 for p in sorted((R/'visualize').glob(pattern)):
  with Image.open(p) as im:im.load();imgs.append({**artifact(p),'size':list(im.size),'decode_ok':True})
videos=[]
for model,e in v['video_verification'].items():
 p=pathlib.Path(e['path']);a={'model':model,**artifact(p)}
 if p.exists():
  probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(p)],capture_output=True,text=True);dec=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],capture_output=True,text=True)
  a.update(sha256_matches=a['sha256']==e['sha256'],probe=json.loads(probe.stdout),probe_returncode=probe.returncode,decode_returncode=dec.returncode,decode_stderr=dec.stderr)
 videos.append(a)
A['visualization']={'representative_frames':[{**samples[i],'frame':i,'elapsed_seconds':(samples[i]['timestamp_us']-samples[0]['timestamp_us'])/1e6} for i in [0,19,38]],'browser_token_time_checks':all(c['token']==f"{samples[c['frame']]['sample_token']} | {samples[c['frame']]['timestamp_us']} us" for c in b['checks']),'browser_existing_checks':b['checks'],'rerun_existing_checks':rr['checks'],'screenshots_and_stills':imgs,'fresh_CPU_video_decode':videos,'rrd':artifact(pathlib.Path(v['rrd_path'])),'rrd_sha256_expected':v['rrd_sha256'],'occupancy_slice':v['occupancy_bev_slice'],'axes_contract':'grid XYZ current LIDAR_TOP; center=min+(index+0.5)*0.5; transform lidar2ego; display horizontal=-ego_y, vertical=-ego_x; z index6=-1.75m','cleanup_code_evidence':'deployment_visualize.py lines150-188: reset history at scene boundary; rr.Clear recursive model/BEV and per-camera overlays each frame; only current IDs re-logged; stored trail truncated to last12 observations. This source audit is not a new browser runtime test.','cleanup_limitation':'History dictionary retains inactive IDs until scene boundary; reappearing ID reconnects stored observations, with no elapsed-time gap test. Stale drawn entities cleared every frame.','source':artifact(ROOT/'perception_lab/tools/deployment_visualize.py')}
if (OUT/'provenance_recovery.json').exists():A['gaussianworld']['tool_provenance_reconciliation']=j(OUT/'provenance_recovery.json')
(OUT/'audit.json').write_text(json.dumps(A,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'sparse4d':{k:{a:v for a,v in s.items() if isinstance(v,(bool,int)) or a=='metrics'} for k,s in A['sparse4d'].items()},'grid_count':len(grids),'grid_checks_pass':all(x['sha256_matches'] and x['class_counts_match'] and x['history_matches'] and x['timestamp_matches'] and x['shape']==[200,200,16] for x in grids),'provenance_mismatches':[x for x in pa if not x['matches']],'GT_matches':matches,'image_count':len(imgs),'video_decode':[{'model':x['model'],'frames':x.get('probe'),'rc':x.get('decode_returncode')} for x in videos]},indent=2))
