"""Thirty AB/BA pairs: nine real NMS inputs and complete 39/81-frame GRAE.
No learned-module hooks; both sides use the same original Torch/model/process.
"""
import time
COLD_START=time.perf_counter()
import torch
torch.cuda.init()
import argparse,csv,hashlib,json,os,resource,subprocess,sys,threading
from pathlib import Path
import numpy as np
from tracking_gpu_kernel_20261013 import GeometryNMS,REPORT,ROOT,ART
from tracking_gpu_geometry_20261013 import grae,make_boxes,pack,cpu_nms
from tracking_gpu_inputs_20261013 import paths,SPECS,sha
import mot_3d.preprocessing as preprocessing
sys.path.insert(0,str(grae.SOURCE));sys.path.insert(0,str(grae.DEPENDENCY))
OUT=REPORT/'benchmark';OUT.mkdir(exist_ok=True)

def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def snapshot():
 def query(args):
  p=subprocess.run(['nvidia-smi']+args,capture_output=True,text=True,timeout=10)
  if p.returncode:raise RuntimeError(p.stderr)
  return p.stdout.strip()
 apps=query(['--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'])
 foreign=[line for line in apps.splitlines() if line and int(line.split(',')[0])!=os.getpid()]
 return dict(time=time.time(),compute_apps=apps,external_compute=foreign,gpu=query(['--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits']),rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())

class Monitor:
 def __init__(self):self.rows=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True)
 def run(self):
  while not self.stop.is_set():
   try:self.rows.append(snapshot())
   except Exception as exc:self.rows.append({'error':str(exc),'external_compute':['monitor_failure']})
   self.stop.wait(2)
 def __enter__(self):self.thread.start();return self
 def __exit__(self,*args):self.stop.set();self.thread.join()

def summary(rows,a,b):
 def stats(vals):return dict(mean=float(np.mean(vals)),p50=float(np.percentile(vals,50)),p95=float(np.percentile(vals,95)))
 result={'cpu':stats([x[a] for x in rows]),'gpu':stats([x[b] for x in rows]),'wins':sum(x[b]<x[a] for x in rows),'rounds':len(rows)}
 result['mean_saving_fraction']=1-result['gpu']['mean']/result['cpu']['mean']
 result['orders']={order:{'cpu':stats([x[a] for x in rows if x['order']==order]),'gpu':stats([x[b] for x in rows if x['order']==order])} for order in ['AB','BA']}
 result['adoption_gate']=len(rows)==30 and result['wins']>=24 and result['gpu']['p95']<=result['cpu']['p95'] and result['mean_saving_fraction']>=.05 and all(v['gpu']['mean']<v['cpu']['mean'] for v in result['orders'].values())
 return result

def csv_rows(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

def real_representatives():
 result=[]
 for split in SPECS:
  dp,pp=paths(split);raw=json.loads(dp.read_text())['results'];packets=json.loads(pp.read_text());groups={}
  for p in packets:groups.setdefault(p['scene_token'],[]).append(p)
  for scene,items in groups.items():
   for j in [0,len(items)//2,len(items)-1]:
    p=items[j];b,c=make_boxes(raw[p['sample_token']]);result.append((split,p,b,c))
 return result

def resident(candidate,b,c):
 geo,order,eligible=pack(b,c);n=len(b)
 g=np.zeros((256,12),np.float64);g[:n]=geo;e=np.zeros((256,256),np.int32);e[:n,:n]=eligible;o=np.zeros(256,np.int64);o[:n]=order
 return (torch.from_numpy(g).cuda(),torch.from_numpy(e).cuda(),torch.tensor([n],dtype=torch.int32,device='cuda'),torch.empty((256,256),dtype=torch.float64,device='cuda'),torch.from_numpy(o).cuda(),torch.empty(257,dtype=torch.int64,device='cuda'))

def kernels(candidate,tensors):
 G,E,N,I,O,K=tensors;a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);c=torch.cuda.Event(enable_timing=True)
 a.record();candidate.launch('overlap',G,E,N,I);b.record();candidate.launch('suppress',I,O,N,K);c.record();c.synchronize()
 return dict(overlap_ms=a.elapsed_time(b),suppress_ms=b.elapsed_time(c),total_ms=a.elapsed_time(c))

def micro(candidate):
 allrows=[];segments=[];results=[]
 for split,p,boxes,classes in real_representatives():
  ref=cpu_nms(boxes,classes,.1);tensors=resident(candidate,boxes,classes)
  for _ in range(3):cpu_nms(boxes,classes,.1);candidate.nms(boxes,classes);kernels(candidate,tensors)
  rows=[]
  for r in range(30):
   values={};order='AB' if r%2==0 else 'BA'
   for mode in order:
    torch.cuda.synchronize();start=time.perf_counter();value=cpu_nms(boxes,classes,.1) if mode=='A' else candidate.nms(boxes,classes);torch.cuda.synchronize();values[mode]=(time.perf_counter()-start)*1000
    if value!=ref:raise ValueError('micro NMS mismatch')
   kt=kernels(candidate,tensors);row=dict(split=split,sample_token=p['sample_token'],scene_token=p['scene_token'],boxes=len(boxes),round=r,order=order,cpu_nms_wall_ms=values['A'],gpu_nms_full_wall_ms=values['B'],**kt);rows.append(row);allrows.append(row)
  # Separately instrument transfers. These events do not pollute acceptance trials.
  for r in range(30):
   geo,o,e=pack(boxes,classes);g=np.zeros((256,12),np.float64);g[:len(boxes)]=geo;el=np.zeros((256,256),np.int32);el[:len(boxes),:len(boxes)]=e;order=np.zeros(256,np.int64);order[:len(boxes)]=o
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);c=torch.cuda.Event(enable_timing=True)
   a.record();G=torch.from_numpy(g).cuda();E=torch.from_numpy(el).cuda();O=torch.from_numpy(order).cuda();N=torch.tensor([len(boxes)],dtype=torch.int32,device='cuda');b.record();b.synchronize();h2d=a.elapsed_time(b)
   candidate.launch('overlap',G,E,N,tensors[3]);a.record();matrix=tensors[3].cpu();b.record();b.synchronize();matrix_d2h=a.elapsed_time(b)
   candidate.launch('suppress',tensors[3],O,N,tensors[5]);a.record();keep=tensors[5].cpu();b.record();b.synchronize()
   segments.append(dict(sample_token=p['sample_token'],round=r,h2d_ms=h2d,matrix_d2h_ms=matrix_d2h,indices_d2h_ms=a.elapsed_time(b)))
  results.append(dict(split=split,sample_token=p['sample_token'],scene_token=p['scene_token'],boxes=len(boxes),paired=summary(rows,'cpu_nms_wall_ms','gpu_nms_full_wall_ms'),kernel_events={k:{'mean':float(np.mean([x[k] for x in rows])),'p50':float(np.percentile([x[k] for x in rows],50)),'p95':float(np.percentile([x[k] for x in rows],95))} for k in ['overlap_ms','suppress_ms','total_ms']}))
  print('micro',split,p['sample_token'],results[-1]['paired']['mean_saving_fraction'],flush=True)
 write(OUT/'micro.json',dict(status='passed',warmups=3,rows=allrows,transfer_rows=segments,representatives=results,kernel_boundary='CUDA events on resident geometry/eligible/order; no copies, CPU sorting, boundary checks or Python reconstruction. No fake CPU GPU-kernel comparator; CPU reference reported only as wall-time full NMS.',preprocessing_boundary='original CPU BBox inputs to consumable NMS indices/types, including all sorting/coarse/packing/H2D/sync/full matrix D2H/CPU boundary recheck/GPU suppression/indices D2H',transfer_boundary='separate instrumented pageable transfers, includes stream gaps; not additive to acceptance times',actual_calls=candidate.calls,boundary_pairs=candidate.boundary_pairs))
 csv_rows(OUT/'representative_pairs.csv',allrows);csv_rows(OUT/'transfer_samples.csv',segments)

def full_run(model,candidate,split,mode):
 preprocessing.nms=cpu_nms if mode=='A' else candidate.nms
 dp,pp=paths(split)
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
 raw=json.loads(dp.read_text())['results'];packets=json.loads(pp.read_text());groups={}
 for p in packets:
  frame=grae.frame_arrays(raw[p['sample_token']],p);groups.setdefault(frame['scene_token'],[]).append(frame)
 predictions={}
 for scene,frames in groups.items():
  output=grae.run_sequence(model,frames,'cuda:0')
  for boxes in output.values():
   for b in boxes:b['tracking_id']=scene+':'+b['tracking_id']
  predictions.update(output)
 torch.cuda.synchronize();elapsed=(time.perf_counter()-started)*1000
 return predictions,elapsed,dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def full(candidate):
 grae.OUT=OUT/'model';start=time.perf_counter();model=grae.load_model('cuda:0');torch.cuda.synchronize();load_ms=(time.perf_counter()-start)*1000
 if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()):raise ValueError('Diagnostic hooks active')
 torch.backends.cudnn.benchmark=False;torch.manual_seed(0);np.random.seed(0)
 reference={split:json.loads((REPORT/'native_compare/grae/frozen_reference'/split/'tracking.json').read_text())['results'] for split in SPECS}
 for split in SPECS:
  for mode in ['A','B']:
   for _ in range(2):
    result,_,_=full_run(model,candidate,split,mode)
    if result!=reference[split]:raise ValueError('Warmup reference mismatch')
 print('full warmups passed',flush=True)
 rows=[];rejected=[]
 for r in range(30):
  for split in SPECS:
   attempt=0
   while True:
    before=snapshot()
    if before['external_compute']:raise RuntimeError('GPU contested before trial: '+str(before))
    order='AB' if r%2==0 else 'BA';values={};mem={};outputs={}
    with Monitor() as monitor:
     for mode in order:outputs[mode],values[mode],mem[mode]=full_run(model,candidate,split,mode)
    after=snapshot();samples=[before]+monitor.rows+[after]
    row=dict(split=split,round=r,order=order,cpu_full_ms=values['A'],gpu_full_ms=values['B'],cpu_output_sha256=digest(outputs['A']),gpu_output_sha256=digest(outputs['B']),cpu_peak_allocated=mem['A']['peak_allocated'],gpu_peak_allocated=mem['B']['peak_allocated'],cpu_peak_reserved=mem['A']['peak_reserved'],gpu_peak_reserved=mem['B']['peak_reserved'],cpu_rss_peak_kib=mem['A']['rss_peak_kib'],gpu_rss_peak_kib=mem['B']['rss_peak_kib'])
    if outputs['A']!=outputs['B'] or outputs['A']!=reference[split]:raise ValueError('Full paired trajectory mismatch')
    write(OUT/f'monitor_{r:02d}_{split}_{attempt}.json',samples)
    if any(x.get('external_compute') for x in samples):
     rejected.append(dict(**row,reason='external_compute_or_monitor_error',attempt=attempt));write(OUT/'rejected_rounds.json',rejected);attempt+=1
     if attempt>=3:raise RuntimeError('Persistent external GPU contention; resume after release')
     continue
    rows.append(row);write(OUT/'full_pairs_partial.json',rows);print('full',r,split,order,round(values['A'],2),round(values['B'],2),flush=True);break
 summaries={s:summary([x for x in rows if x['split']==s],'cpu_full_ms','gpu_full_ms') for s in SPECS}
 combined=[dict(round=r,order='AB' if r%2==0 else 'BA',cpu_full_ms=sum(x['cpu_full_ms'] for x in rows if x['round']==r),gpu_full_ms=sum(x['gpu_full_ms'] for x in rows if x['round']==r)) for r in range(30)]
 result=dict(status='passed',rounds_per_split=30,frames_per_round={'scene0061':39,'mini_val':81},warmups_per_mode_per_split=2,boundary='warm frozen JSON and packets read -> exact native reader or TileLang NMS -> original GRAE H2D/network/association/D2H and consumable Python tracks; disk output, evaluation, correctness hashing excluded. Whole sequences used; reset at each true scene start.',hooks=False,model_load_ms=load_ms,process_import_cuda_init_ms=IMPORT_MS,compile='precompiled separately, see kernelbench/compile_manifest.json',rows=rows,combined_rounds=combined,summary=summaries,combined_summary=summary(combined,'cpu_full_ms','gpu_full_ms'),candidate_adopted=all(v['adoption_gate'] for v in summaries.values()) and summary(combined,'cpu_full_ms','gpu_full_ms')['adoption_gate'],rejected_trials=rejected,source_sha256=sha(__file__),runtime_sources={name:sha(ROOT/'tools'/name) for name in ['tracking_gpu_kernel_20261013.py','tracking_gpu_geometry_20261013.py','grae3dmot_2025.py']},kernel_manifest=json.loads((ART/'manifest.json').read_text()),input_sha256={s:sha(paths(s)[0]) for s in SPECS},actual_kernel_calls=candidate.calls,cpu_boundary_pairs=candidate.boundary_pairs,device=torch.cuda.get_device_name(),memory_scope='Torch peak allocated/reserved per complete split; process total GPU used_memory snapshots every2s and boundaries; CPU lifetime ru_maxrss high-water mark, not incremental allocations')
 write(OUT/'full.json',result);csv_rows(OUT/'full_pairs.csv',rows);print(json.dumps(summaries,indent=2),flush=True)

IMPORT_MS=(time.perf_counter()-COLD_START)*1000
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['micro','full','all'],default='all');a=p.parse_args()
 if json.loads((REPORT/'kernel_correctness.json').read_text())['status']!='passed':raise RuntimeError('Correctness prerequisite')
 candidate=GeometryNMS()
 if a.phase in ['micro','all']:
  initial=snapshot()
  if initial['external_compute']:raise RuntimeError('GPU contested before micro')
  with Monitor() as monitor:micro(candidate)
  samples=[initial]+monitor.rows+[snapshot()];write(OUT/'micro_gpu_monitor.json',samples)
  if any(x.get('external_compute') for x in samples):
   target=OUT/'micro.json';report=json.loads(target.read_text());report['status']='rejected_external_competition';write(OUT/('micro_rejected_'+str(int(time.time()))+'.json'),report);target.unlink();raise RuntimeError('Micro trial set rejected; full set must be rerun after contention clears')
 if a.phase in ['full','all']:full(candidate)
