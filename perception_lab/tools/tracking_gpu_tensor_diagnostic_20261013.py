"""Limited analytical PyTorch reference plus actual CUDA profiler evidence."""
import torch
torch.cuda.init()
import json,hashlib
import numpy as np
from tracking_gpu_kernel_20261013 import GeometryNMS,REPORT
from tracking_gpu_geometry_20261013 import BBox,pack,cpu_nms
from tracking_gpu_inputs_20261013 import sha

def main():
 candidate=GeometryNMS();boxes=[]
 for x,y,z,l,w,h in [(0,0,0,4,2,2),(1,0,0,4,2,10),(1,0,6,4,2,1),(8,0,0,3,3,2)]:
  b=BBox(x=x,y=y,z=z,l=l,w=w,h=h,o=0);b.s=.9;boxes.append(b)
 classes=np.array([0,0,0,1],np.int64);geo,order,eligible=pack(boxes,classes)
 a=torch.tensor([[b.x,b.y,b.z,b.l,b.w,b.h] for b in boxes],dtype=torch.float64,device='cuda')
 dx=(torch.minimum(a[:,0,None]+a[:,3,None]/2,a[None,:,0]+a[None,:,3]/2)-torch.maximum(a[:,0,None]-a[:,3,None]/2,a[None,:,0]-a[None,:,3]/2)).clamp_min(0)
 dy=(torch.minimum(a[:,1,None]+a[:,4,None]/2,a[None,:,1]+a[None,:,4]/2)-torch.maximum(a[:,1,None]-a[:,4,None]/2,a[None,:,1]-a[None,:,4]/2)).clamp_min(0)
 height=torch.minimum((a[:,2,None]+a[:,5,None]/2)-(a[None,:,2]-a[None,:,5]/2),(a[None,:,2]+a[None,:,5]/2)-(a[:,2,None]-a[:,5,None]/2)).clamp_min(0)
 inter=dx*dy*height;volume=a[:,3]*a[:,4]*a[:,5];reference=inter/(volume[:,None]+volume[None,:]-inter+1e-5);reference*=torch.from_numpy(eligible).cuda()
 g=np.zeros((256,12),np.float64);g[:4]=geo;e=np.zeros((256,256),np.int32);e[:4,:4]=eligible
 G=torch.from_numpy(g).cuda();E=torch.from_numpy(e).cuda();N=torch.tensor([4],dtype=torch.int32,device='cuda');I=torch.empty((256,256),dtype=torch.float64,device='cuda')
 candidate.launch('overlap',G,E,N,I);actual=I[:4,:4];error=(actual-reference).abs()
 if not torch.all(error<=1e-8+1e-7*reference.abs()):raise ValueError('Analytical aligned reference mismatch')
 expected=cpu_nms(boxes,classes,.1)
 with torch.autograd.profiler.profile(use_cuda=True) as prof:
  value=candidate.nms(boxes,classes);torch.cuda.synchronize()
 if value!=expected:raise ValueError('Profiler NMS mismatch')
 prof.export_chrome_trace(str(REPORT/'kernelbench/cuda_trace.json'))
 events=[dict(name=x.name,cpu_time_us=x.cpu_time_total,cuda_time_us=x.cuda_time_total) for x in prof.function_events if x.cuda_time_total>0]
 result=dict(status='passed',scope='PyTorch analytical FP64 reference only axis-aligned boxes, original nonstandard height; NOT general rotated IoU or NMS reference. General reference remains original CPU Shapely.',max_abs=float(error.max()),reference_sha256=hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest(),output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),actual_calls=candidate.calls,profiler_events=events,trace='kernelbench/cuda_trace.json',trace_note='Diagnostic only; profiler overhead excluded from all paired acceptance measurements. No hardware occupancy or memory-throughput counters claimed.',source_sha256=sha(__file__))
 (REPORT/'kernelbench/pytorch_axis_reference.json').write_text(json.dumps(result,indent=2)+'\n');print('Analytical PyTorch reference passed')
if __name__=='__main__':main()
