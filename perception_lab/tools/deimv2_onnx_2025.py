"""ONNX Runtime FP32 validation against measured native COCO output, CUDA only."""
import argparse,gzip,hashlib,json,time,os,resource
os.environ["NVIDIA_TF32_OVERRIDE"]="0"
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 from deimv2_backend_identity_2025 import verify_baseline
 identity=verify_baseline(check_predictions=True)
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);a=ap.parse_args()
 import torch,torchvision.transforms as T
 import onnxruntime as ort
 from PIL import Image
 from deimv2_2025 import coco_rows
 out=ROOT/'reports/perception_2025_20261013/deimv2/onnx';out.mkdir(exist_ok=True);write(out/'status.json',dict(status='running',images=0));model=ROOT/'checkpoints/perception_2025/deimv2/deimv2_s.onnx';native=ROOT/'runs/perception_2025/deimv2/coco_predictions.json'
 options=ort.SessionOptions();options.enable_profiling=True;options.profile_file_prefix=str(ROOT/'runs/perception_2025/deimv2/ort_profile');options.intra_op_num_threads=4
 session=ort.InferenceSession(str(model),sess_options=options,providers=[('CUDAExecutionProvider',{'use_tf32':0} if tuple(map(int,ort.__version__.split('.')[:2])) >= (1,18) else {}),'CPUExecutionProvider']);assert session.get_providers()[0]=='CUDAExecutionProvider'
 write(out/'io.json',dict(inputs=[dict(name=x.name,type=x.type,shape=x.shape) for x in session.get_inputs()],outputs=[dict(name=x.name,type=x.type,shape=x.shape) for x in session.get_outputs()],runtime=ort.__version__,providers=session.get_providers(),provider_options=session.get_provider_options(),tf32_override=os.environ["NVIDIA_TF32_OVERRIDE"],model_sha256=sha(model)))
 trans=T.Compose([T.Resize((640,640)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]);inventory=json.loads((ROOT/'reports/perception_2025_20261013/inputs/coco_files.json').read_text());cats=sorted(x['id'] for x in json.loads((ROOT/'data/coco/annotations/instances_val2017.json').read_text())['categories'])
 def infer(item):
  start=time.perf_counter();im=Image.open(item['file']).convert('RGB');x=trans(im).unsqueeze(0).numpy();orig=np.array([im.size],dtype=np.int64);prep=time.perf_counter();labels,boxes,scores=session.run(None,{'images':x,'orig_target_sizes':orig});end=time.perf_counter()
  return labels[0],boxes[0],scores[0],dict(e2e_ms=(end-start)*1000,preprocess_ms=(prep-start)*1000,forward_h2d_d2h_ms=(end-prep)*1000)
 for _ in range(5):infer(inventory[0])
 profile=Path(session.end_profiling());events=json.loads(profile.read_text());providers={}
 for e in events:
  provider=e.get('args',{}).get('provider')
  if provider:providers[provider]=providers.get(provider,0)+1
 assert providers.get('CUDAExecutionProvider',0)>0,'No measured CUDA kernels'
 pred=[];times=[];gpu_used_peak=0;selected=inventory[:a.limit] if a.limit else inventory
 for i,item in enumerate(selected):
  l,b,s,t=infer(item);free,total=torch.cuda.mem_get_info();gpu_used_peak=max(gpu_used_peak,total-free);pred.extend(coco_rows(item['id'],l,b,s,cats));times.append({'image_id':item['id'],**t})
  if i%100==0:print('ONNX',i,len(selected),flush=True)
 run=ROOT/'runs/perception_2025/deimv2';write(run/'onnx_predictions.json',pred)
 with gzip.open(out/'predictions.json.gz','wt') as f:json.dump(pred,f)
 write(out/'image_ids.json',[r['id'] for r in selected]);write(out/'timings.json',times)
 import subprocess,sys
 with (out/'official_eval.log').open('w') as log:subprocess.run([str(ROOT/'envs/nuscenes_eval/bin/python'),str(ROOT/'tools/coco_eval_2025.py'),'--annotations',str(ROOT/'data/coco/annotations/instances_val2017.json'),'--predictions',str(run/'onnx_predictions.json'),'--image-ids',str(out/'image_ids.json'),'--out',str(out/'official_eval')],check=True,stdout=log,stderr=subprocess.STDOUT)
 metrics=json.loads((out/'official_eval/metrics.json').read_text());reference=json.loads((out.parent/'official_eval/metrics.json').read_text());refby={}
 for b in json.loads(native.read_text()):refby.setdefault(b['image_id'],[]).append(b)
 predby={}
 for b in pred:predby.setdefault(b['image_id'],[]).append(b)
 comparisons=[]
 from scipy.optimize import linear_sum_assignment
 for item in selected:
  p=predby[item['id']];r=refby[item['id']];maxbox=0.;maxscore=0.;unmatched=0
  for cat in cats:
   aa=[x for x in r if x['category_id']==cat];bb=[x for x in p if x['category_id']==cat]
   if not aa or not bb:unmatched+=len(aa)+len(bb);continue
   A=np.array([x['bbox'] for x in aa]);B=np.array([x['bbox'] for x in bb]);dist=np.linalg.norm(A[:,None]-B[None],axis=2);ii,jj=linear_sum_assignment(dist);unmatched+=abs(len(aa)-len(bb));maxbox=max(maxbox,float(np.max(np.abs(A[ii]-B[jj]))));maxscore=max(maxscore,max(abs(aa[i]['score']-bb[j]['score']) for i,j in zip(ii,jj)))
  comparisons.append(dict(image_id=item['id'],max_box_component_px=maxbox,max_score_difference=maxscore,unmatched_by_class=unmatched))
 write(out/'geometry_diagnostics.json',comparisons)
 same_scope=[r['id'] for r in selected]==json.loads((out.parent/'image_ids.json').read_text())
 drop=reference['AP']-metrics['AP'] if same_scope else None
 write(out/'status.json',dict(status='partial',images=len(selected),identity=identity,gpu_device_used_peak_bytes=gpu_used_peak,gpu_memory_scope='device total incl other resident processes; sampled after inference, not isolated allocator peak',ram_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,metrics=metrics,ap_drop=drop,ap_budget_pass=drop<=.005 if drop is not None else None,ap_comparison_same_image_ids=same_scope,geometry='see complete diagnostics; candidate identities not assumed by row',provider_kernel_counts=providers,profiling_enabled='warmup only; ended before measured sequence',timing_limit='unpaired sequence, not optimization benchmark',predictions_sha256=sha(run/'onnx_predictions.json')))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(ROOT/'reports/perception_2025_20261013/deimv2/onnx/status.json',dict(status='failed',error=repr(exc)))
  raise
