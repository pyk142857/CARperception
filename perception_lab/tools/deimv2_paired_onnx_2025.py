"""Serial 30-round native/ORT paired end-to-end timing in isolated persistent workers."""
import argparse,contextlib,hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ['NVIDIA_TF32_OVERRIDE']='0'
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/perception_2025_20261013/deimv2/onnx'
def worker(backend):
 cold=time.perf_counter()
 with contextlib.redirect_stdout(sys.stderr):
  import numpy as np,torch,torchvision.transforms as T,resource
  from PIL import Image
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  trans=T.Compose([T.Resize((640,640)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
  if backend=='native':
   c=json.loads((ROOT/'configs/perception_2025.yaml').read_text())['models']['deimv2'];source=ROOT/c['source'];sys.path.insert(0,str(source));from engine.core import YAMLConfig
   cfg=YAMLConfig(str(source/c['config']),resume=str(ROOT/c['checkpoint']));cfg.yaml_cfg['DINOv3STAs']['weights_path']=None
   ck=torch.load(ROOT/c['checkpoint'],map_location='cpu',weights_only=False);model=cfg.model;model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model'],strict=True);model=model.eval().cuda().deploy();post=cfg.postprocessor.deploy().cuda()
  else:
   import onnxruntime as ort
   opt=ort.SessionOptions();opt.intra_op_num_threads=4
   session=ort.InferenceSession(str(ROOT/'checkpoints/perception_2025/deimv2/deimv2_s.onnx'),sess_options=opt,providers=['CUDAExecutionProvider','CPUExecutionProvider']);assert session.get_providers()[0]=='CUDAExecutionProvider'
  def infer(item):
   start=time.perf_counter()
   with Image.open(item['file']) as im:im=im.convert('RGB')
   x=trans(im).unsqueeze(0);size=torch.tensor([im.size],dtype=torch.int64);pre=time.perf_counter()
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   if backend=='native':
    gx=x.cuda();gs=size.cuda();torch.cuda.synchronize();transfer=time.perf_counter();a.record()
    with torch.inference_mode():ys=post(model(gx),gs)
    b.record();b.synchronize();decode_start=time.perf_counter();ys=[v.cpu().numpy() for v in ys];finish=time.perf_counter();gpu=a.elapsed_time(b)
    sections=dict(h2d_wall_ms=(transfer-pre)*1000,forward_post_gpu_ms=gpu,
                  forward_post_wall_ms=(decode_start-transfer)*1000,decode_d2h_wall_ms=(finish-decode_start)*1000)
   else:
    torch.cuda.synchronize();a.record();ys=session.run(None,{'images':x.numpy(),'orig_target_sizes':size.numpy()});b.record();b.synchronize();finish=time.perf_counter()
    sections=dict(ort_h2d_forward_d2h_wall_ms=(finish-pre)*1000,cuda_interval_ms=a.elapsed_time(b))
   free,total=torch.cuda.mem_get_info()
   return dict(image_id=item['id'],e2e_ms=(finish-start)*1000,read_preprocess_ms=(pre-start)*1000,**sections,input_sha256=hashlib.sha256(x.numpy().tobytes()+size.numpy().tobytes()).hexdigest(),output_sha256=hashlib.sha256(b''.join(y.tobytes() for y in ys)).hexdigest(),finite=all(np.isfinite(y).all().item() for y in ys),device_used_bytes=total-free,ram_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  inv=json.loads((ROOT/'reports/perception_2025_20261013/inputs/coco_files.json').read_text());items=[inv[i] for i in [0,len(inv)//2,len(inv)-1]]
  for item in items:
   for _ in range(3):infer(item)
 print(json.dumps(dict(ready=True,backend=backend,cold_load_and_warmup_ms=(time.perf_counter()-cold)*1000)),flush=True)
 for line in sys.stdin:
  req=json.loads(line)
  if req.get('stop'):break
  with contextlib.redirect_stdout(sys.stderr):res=infer(items[req['index']])
  print(json.dumps(res),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['native','onnx']);a=ap.parse_args()
 if a.worker:return worker(a.worker)
 import statistics
 from deimv2_backend_identity_2025 import verify_baseline
 identity=verify_baseline()
 c=json.loads((ROOT/'configs/perception_2025.yaml').read_text());procs={};logs=[];ready={};rounds=[];OUT.mkdir(parents=True,exist_ok=True);status=OUT/'paired_status.json';status.write_text(json.dumps(dict(status='running')))
 try:
  for backend,key in [('native','deimv2'),('onnx','onnx')]:
   log=(OUT/(backend+'_paired_worker.log')).open('w');logs.append(log)
   p=subprocess.Popen([str(ROOT/c['environments'][key]),str(Path(__file__).resolve()),'--worker',backend],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=dict(os.environ,OMP_NUM_THREADS='8'));procs[backend]=p
   ready[backend]=json.loads(p.stdout.readline());assert ready[backend]['ready']
  for i in range(30):
   row=dict(round=i,order=['native','onnx'] if i%2==0 else ['onnx','native'])
   for backend in row['order']:
    p=procs[backend];p.stdin.write(json.dumps(dict(index=i%3))+'\n');p.stdin.flush();row[backend]=json.loads(p.stdout.readline());assert row[backend]['finite']
   assert row['native']['input_sha256']==row['onnx']['input_sha256'],'paired input bytes differ'
   rounds.append(row)
  def stats(values):
   v=sorted(values)
   def pct(q):
    x=(len(v)-1)*q;j=int(x);return v[j]+(v[min(j+1,len(v)-1)]-v[j])*(x-j)
   return dict(mean=statistics.mean(v),p50=pct(.5),p95=pct(.95),std=statistics.pstdev(v))
  summary={b:{key:stats([r[b][key] for r in rounds]) for key in rounds[0][b] if key.endswith('_ms')} for b in procs}
  savings=[r['native']['e2e_ms']-r['onnx']['e2e_ms'] for r in rounds];orders={str(i):stats(savings[i::2]) for i in [0,1]}
  repeat={b:all(len({r[b]['output_sha256'] for r in rounds if r['round']%3==i})==1 for i in [0,1,2]) for b in procs}
  result=dict(status='passed',identity=identity,rounds=rounds,summary=summary,e2e_saving_ms=stats(savings),order_saving_ms=orders,same_backend_bitwise_repeat=repeat,cold=ready,scope='Both persistent models resident, serial GPU execution, 30 AB/BA pairs, same 3 raw images and identical input bytes. Warm OS file cache; reread/decode/normalize/H2D/forward/postprocess/D2H included. Hashing/IPC excluded.',cuda_interval_note='ORT CUDA event interval encloses host scheduling plus H2D/inference/D2H; not a pure kernel-only timing.',memory_note='Device total includes both workers and unrelated resident allocations; not isolated per-model allocation.',adoption='Native retained unless full COCO and geometry acceptance independently support backend adoption')
  (OUT/'paired_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');status.write_text(json.dumps(dict(status='passed',rounds=30))+'\n')
 except Exception as exc:
  status.write_text(json.dumps(dict(status='failed',error=repr(exc)))+'\n');raise
 finally:
  for p in procs.values():
   if p.poll() is None:
    try:p.stdin.write('{"stop":true}\n');p.stdin.flush();p.wait(timeout=20)
    except Exception:p.terminate();p.wait(timeout=10)
  for f in logs:f.close()
if __name__=='__main__':main()
