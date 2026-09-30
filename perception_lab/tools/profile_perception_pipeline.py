"""Profile real offline entrypoints in isolated outputs, using synchronized phase boundaries."""
import argparse,atexit,json,os,sys,time,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/pipeline_timing';RUNTIME=ROOT/'outputs/pipeline_timing'

PATCHES={
 'vision_infer.py':[
 ('        image = cv2.imread',"        _phase('input_decode')\n        image = cv2.imread"),
 ('        prefix =',"        _phase('adapter_validation')\n        prefix ="),
 ('                r = model.predict',"                _phase('inference_api')\n                r = model.predict"),
 ('                native =',"                _phase('postprocess_export')\n                native ="),
 ('                depth = model.infer_image',"                _phase('inference_api')\n                depth = model.infer_image"),
 ('                if depth.shape',"                _phase('postprocess_export')\n                if depth.shape"),
 ("    (out / 'predictions.json')", "    _phase('finalize')\n    (out / 'predictions.json')")],
 'segformer_infer.py':[
 ('        with torch.inference_mode():',"        _phase('inference_api_including_input')\n        with torch.inference_mode():"),
 ('        seg =',"        _phase('postprocess_export')\n        seg ="),
 ("    (out/'labels.json')", "    _phase('finalize')\n    (out/'labels.json')")],
 'lidar_smoke.py':[
 (' data=pipeline(inputs)'," _phase('input_preprocess')\n data=pipeline(inputs)"),
 (' with torch.inference_mode():'," _phase('inference_api')\n with torch.inference_mode():"),
 (' pred=result.pred_instances'," _phase('postprocess_export')\n pred=result.pred_instances"),
 (' import matplotlib\n'," _phase('render_export')\n import matplotlib\n")],
 'maptr_mini.py':[
 ("        lidar = packet", "        _phase('input_preprocess')\n        lidar = packet"),
 ('        with torch.no_grad():',"        _phase('inference_api')\n        with torch.no_grad():"),
 ("        xy = pred", "        _phase('postprocess')\n        xy = pred"),
 ('    validate_frames(results, packets)',"    _phase('finalize')\n    validate_frames(results, packets)")],
 'lidarseg_mini.py':[
 ('        path=Path(packet',"        _phase('input_preprocess')\n        path=Path(packet"),
 ('        with torch.inference_mode():',"        _phase('inference_api_with_point_decode')\n        with torch.inference_mode():"),
 ('        del logits',"        _phase('evaluation_export')\n        del logits"),
 ('    union=hist.sum',"    _phase('finalize')\n    union=hist.sum")],
 'track_mini.py':[
 ('  for rec,packet in zip(records,packets):',"  _phase('tracking_with_coordinate_adapters')\n  for rec,packet in zip(records,packets):"),
 (' tracks,times=run();repeated,_=run()'," _phase('tracking_with_coordinate_adapters')\n tracks,times=run();repeated,_=run()\n _phase('postprocess_export')"),
 (' history={};images=[];bundles=[]'," _phase('render_export')\n history={};images=[];bundles=[]")],
 'rerun_mini.py':[
 ('    for i, (packet, pred, tracked)',"    _phase('recording_build')\n    for i, (packet, pred, tracked)"),
 ('    rr.get_global_data_recording().flush()',"    _phase('recording_flush')\n    rr.get_global_data_recording().flush()"),
 ("    summary = dict(","    _phase('finalize')\n    summary = dict(")],
}

def worker(script,output,args):
 started=time.perf_counter();last=started;label='imports_setup_weights';events=[]
 def sync():
  torch=sys.modules.get('torch')
  if torch is not None and torch.cuda.is_initialized():torch.cuda.synchronize()
 def phase(new):
  nonlocal last,label
  sync();now=time.perf_counter();events.append(dict(phase=label,seconds=now-last));last=now;label=new
 def finish():
  phase('end');Path(output).write_text(json.dumps(dict(events=events,instrumented_seconds=time.perf_counter()-started),indent=2)+'\n')
 atexit.register(finish)
 warm='--resident-probe' in args
 if warm:args.remove('--resident-probe')
 source=ROOT/'tools'/script;code=source.read_text()
 for old,new in PATCHES[script]:
  if old not in code:raise ValueError('Instrumentation anchor missing: '+old)
  code=code.replace(old,new)
 if warm:
  assert script=='lidar_smoke.py'
  anchor=" with torch.inference_mode(): result=model.test_step(pseudo_collate([data]))[0]"
  probe=" with torch.inference_mode():\n  _phase('resident_warmup')\n  for _ in range(3): model.test_step(pseudo_collate([data]))\n  for _ in range(20):\n   _phase('resident_inference')\n   model.test_step(pseudo_collate([data]))\n  _phase('original_inference_after_probe')\n"
  assert anchor in code;code=code.replace(anchor,probe+anchor)
 sys.path.insert(0,str(ROOT/'tools'));sys.argv=[str(source)]+args
 exec(compile(code,str(source),'exec'),dict(__name__='__main__',__file__=str(source),_phase=phase))

def main():
 if '--worker' in sys.argv:
  i=sys.argv.index('--worker');worker(sys.argv[i+1],sys.argv[i+2],sys.argv[i+3:]);return
 REPORT.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(parents=True,exist_ok=True)
 if '--resident-probes' in sys.argv:
  probes=[]
  for model in ['centerpoint','pointpillars']:
   name=model+'_resident';cmd=[str(ROOT/'envs/mmdet3d/bin/python'),str(Path(__file__).resolve()),'--worker','lidar_smoke.py',str(REPORT/(name+'_phases.json')),'--model',model,'--packets',str(ROOT/'manifests/frame_packets.json'),'--sample-index','0','--out',str(RUNTIME/name),'--resident-probe']
   start=time.perf_counter()
   with (REPORT/(name+'.log')).open('w') as f:result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   probes.append(dict(name=name,command=cmd,returncode=result.returncode,wall_seconds=time.perf_counter()-start))
   (REPORT/'resident_jobs.json').write_text(json.dumps(probes,indent=2)+'\n')
   if result.returncode:raise RuntimeError('Resident probe failed: '+model)
  return
 packets=json.loads((ROOT/'manifests/frame_packets.json').read_text());images=[]
 for p in packets[:3]:
  for c,s in p['sensors'].items():
   if c.startswith('CAM_'):images.append(dict(path=s['path'],camera_channel=c,sample_token=p['sample_token'],scene_token=p['scene_token'],timestamp_us=p['timestamp_us']))
 manifest=RUNTIME/'images.json';manifest.write_text(json.dumps(dict(scope='profiling_first_3_frames_6_cameras',images=images)))
 jobs=[]
 for model in ['yolo','depth']:
  jobs.append((model,'vision','vision_infer.py',['--model',model,'--manifest',str(manifest),'--device','cuda:0']))
 jobs.append(('segformer','vision','segformer_infer.py',['--manifest',str(manifest),'--device','cuda:0']))
 for model in ['pointpillars','centerpoint']:
  for i in range(3):jobs.append((model+'_'+str(i),'mmdet3d','lidar_smoke.py',['--model',model,'--packets',str(ROOT/'manifests/frame_packets.json'),'--sample-index',str(i)]))
 jobs.extend([('maptr','maptr','maptr_mini.py',['--limit','3']),('lidarseg','lidarseg_legacy','lidarseg_mini.py',['--limit','3'])])
 # Original tracking entrypoint requires a runs/<run>/<module>/<stage> output layout.
 status=json.loads((ROOT/'results/status.json').read_text());dp=Path(status['modules']['M05']['mini_scene']['log']).parent/'predictions.json'
 jobs.append(('tracking_39','mmdet3d','track_mini.py',['--detections',str(dp)]))
 jobs.append(('rerun_export_39','rerun','rerun_mini.py',['--no-failures']))
 results=[];overall=time.perf_counter()
 for name,env,script,args in jobs:
  out=ROOT/'runs/pipeline_timing'/name/'profile' if name.startswith('tracking') else RUNTIME/name
  if name.startswith('rerun'):out=out/'scene.rrd'
  out.parent.mkdir(parents=True,exist_ok=True)
  cmd=[str(ROOT/'envs'/env/'bin/python'),str(Path(__file__).resolve()),'--worker',script,str(REPORT/(name+'_phases.json'))]+args+['--out',str(out)]
  environment=os.environ.copy();environment['YOLO_CONFIG_DIR']=str(ROOT/'envs/yolo_settings');environment['MPLBACKEND']='Agg'
  start=time.perf_counter()
  with (REPORT/(name+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT)
  row=dict(name=name,script=script,env=env,command=cmd,returncode=p.returncode,wall_seconds=time.perf_counter()-start,source_sha256=hashlib.sha256((ROOT/'tools'/script).read_bytes()).hexdigest());results.append(row)
  (REPORT/'jobs.json').write_text(json.dumps(dict(jobs=results,total_wall_seconds=time.perf_counter()-overall),indent=2)+'\n')
  print(name,p.returncode,round(row['wall_seconds'],3),flush=True)
 print('Completed sequential measurements',flush=True)
if __name__=='__main__':main()
