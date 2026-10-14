"""Actual typed TileLang/NMS validation against pinned CPU semantics."""
import torch
# Initialize original Torch CUDA before importing the legacy NumPy/Numba dependency graph.
torch.cuda.init()
import sys,json,hashlib,time,subprocess
from pathlib import Path
import numpy as np
from tracking_gpu_kernel_20261013 import GeometryNMS,REPORT,ROOT,ART
from tracking_gpu_geometry_20261013 import pack,make_boxes,cpu_nms,BBox,array_hash,grae
from tracking_gpu_inputs_20261013 import paths,SPECS,sha
from mot_3d.utils import iou3d
import mot_3d.preprocessing as preprocessing

def compare_geometry(candidate,boxes,classes):
 n=len(boxes);geo,order,eligible=pack(boxes,classes)
 g=np.zeros((256,12),np.float64);g[:n]=geo;e=np.zeros((256,256),np.int32);e[:n,:n]=eligible
 G=torch.from_numpy(g).cuda();E=torch.from_numpy(e).cuda();N=torch.tensor([n],dtype=torch.int32,device='cuda');O=torch.empty((256,256),dtype=torch.float64,device='cuda')
 candidate.launch('overlap',G,E,N,O);actual=O.cpu().numpy()[:n,:n];reference=np.zeros((n,n));
 for i,j in np.argwhere(eligible):reference[i,j]=iou3d(boxes[i],boxes[j])[1]
 diff=np.abs(actual-reference);budget=1e-8+1e-7*np.abs(reference)
 if not np.all(diff<=budget):raise ValueError('FP64 geometry budget failed: '+str(diff.max()))
 worst=np.unravel_index(np.argmax(diff),diff.shape) if n else (0,0)
 return dict(pairs=int(eligible.sum()),max_abs=float(diff.max()) if n else 0.,max_relative=float(np.max(diff/np.maximum(np.abs(reference),1e-30))) if n else 0.,worst_pair=list(map(int,worst)),input_dtypes=['float64','int32','int32'],output_dtype='float64',input_sha256=hashlib.sha256(g.tobytes()+e.tobytes()).hexdigest(),output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())

def fixtures():
 def box(x=0,y=0,z=0,l=4,w=2,h=2,yaw=0,score=.9):
  b=BBox(x=x,y=y,z=z,l=l,w=w,h=h,o=yaw);b.s=score;return b
 result=[('empty',[],[]),('single',[box()],[0]),('duplicate_ties',[box(),box()],[0,0]),('class_separation',[box(),box()],[0,1]),('touching',[box(),box(x=4)],[0,0]),('z_separation',[box(),box(z=5)],[0,0]),('height_containment_original',[box(h=10),box(h=1)],[0,0]),('negative_grid_boundary',[box(x=-100.1,y=-.1),box(x=-99.9,y=.1)],[0,0]),('pi_rotation',[box(yaw=np.pi-1e-10),box(yaw=-np.pi+1e-10)],[0,0]),('world_offset',[box(x=2000,y=1800),box(x=2000.2,y=1800.1)],[0,0])]
 for n in [63,64,65,199,256]:result.append(('mask_'+str(n),[box(x=20*i,score=.9) for i in range(n)],[0]*n))
 # Solve the original 3D epsilon denominator for overlap area at threshold .1.
 area=.1*(2*4*2*2+1e-5)/(2*1.1);dx=4-area/2
 for delta in [-1e-7,0,1e-7]:result.append(('threshold_'+str(delta),[box(),box(x=dx+delta)],[0,0]))
 rng=np.random.default_rng(20261013)
 b=[box(x=float(rng.uniform(-5,5)),y=float(rng.uniform(-5,5)),h=float(rng.uniform(.5,4)),yaw=float(rng.uniform(-np.pi,np.pi)),score=.9) for _ in range(65)]
 result.append(('random_same_scores',b,[i%3 for i in range(65)]));idx=rng.permutation(65);result.append(('shuffled_same_scores',[b[i] for i in idx],[int(i%3) for i in idx]))
 return result

def main():
 candidate=GeometryNMS();rows=[];fixture_rows=[]
 for name,boxes,classes in fixtures():
  classes=np.asarray(classes,np.int64)
  reference=cpu_nms(boxes,classes,.1) if boxes else ([],[])
  actual=candidate.nms(boxes,classes)
  if reference!=actual:raise ValueError('Fixture NMS mismatch '+name+str((reference,actual)))
  detail=compare_geometry(candidate,boxes,classes) if boxes else {}
  fixture_rows.append(dict(name=name,n=len(boxes),indices_order_types_exact=True,geometry=detail))
 expected={r['sample_token']:r['input_array_sha256'] for r in json.loads((REPORT/'cpu_profile.json').read_text())['rows']}
 for split in SPECS:
  dp,pp=paths(split);raw=json.loads(dp.read_text())['results'];packets=json.loads(pp.read_text());groups={}
  for i,p in enumerate(packets):groups.setdefault(p['scene_token'],[]).append(i)
  representative={ids[j] for ids in groups.values() for j in [0,len(ids)//2,len(ids)-1]}
  for index,p in enumerate(packets):
   b,c=make_boxes(raw[p['sample_token']]);reference=cpu_nms(b,c,.1);actual=candidate.nms(b,c)
   if reference!=actual:raise ValueError('Actual NMS mismatch '+p['sample_token'])
   previous=preprocessing.nms;preprocessing.nms=candidate.nms
   try:arrays=grae.frame_arrays(raw[p['sample_token']],p)
   finally:preprocessing.nms=previous
   digest=array_hash(arrays)
   if digest!=expected[p['sample_token']]:raise ValueError('GRAE input bytes changed')
   detail=compare_geometry(candidate,b,c) if index in representative else None
   rows.append(dict(split=split,index=index,sample_token=p['sample_token'],n=len(b),indices=actual[0],types=list(map(int,actual[1])),indices_order_types_exact=True,model_input_bytes_exact=True,input_array_sha256=digest,geometry=detail))
   if index%20==0:print(split,index,flush=True)
 result=dict(status='passed',frames=len(rows),fixture_count=len(fixture_rows),seed=20261013,fp64_atol=1e-8,fp64_rtol=1e-7,integer_comparison='exact including order and types',actual_kernel_calls=candidate.calls,boundary_cpu_rechecks=candidate.boundary_pairs,nms_calls=candidate.frames,fixtures=fixture_rows,rows=rows,compiled_manifest=json.loads((ART/'manifest.json').read_text()),source_sha256=sha(__file__))
 (REPORT/'kernel_correctness.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),len(fixture_rows),candidate.calls)
if __name__=='__main__':main()
