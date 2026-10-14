"""Real TileLang FP64 convex clipping and GPU greedy NMS; fixed <=256 boxes."""
from pathlib import Path
import argparse,ctypes,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'runs/tracking_gpu_20261013/tilelang_geometry'
REPORT=ROOT/'reports/tracking_gpu_20261013'
CAPACITY=256

def compile_kernels():
 import tilelang
 import tilelang.language as T
 @T.prim_func
 def overlap(G:T.Tensor((256,12),'float64'), E:T.Tensor((256,256),'int32'), N:T.Tensor((1,),'int32'), O:T.Tensor((256,256),'float64')):
  with T.Kernel(512,threads=128) as block:
   tx=T.get_thread_binding()
   i=(block*128+tx)//256
   j=(block*128+tx)%256
   px=T.alloc_local((16,),'float64');py=T.alloc_local((16,),'float64')
   qx=T.alloc_local((16,),'float64');qy=T.alloc_local((16,),'float64')
   count=T.alloc_var('int32',init=4);next_count=T.alloc_var('int32',init=0)
   area=T.alloc_var('float64',init=0)
   ox=T.alloc_var('float64',init=0);oy=T.alloc_var('float64',init=0)
   ax=T.alloc_var('float64',init=0);ay=T.alloc_var('float64',init=0)
   bx=T.alloc_var('float64',init=0);by=T.alloc_var('float64',init=0)
   sx=T.alloc_var('float64',init=0);sy=T.alloc_var('float64',init=0)
   ex=T.alloc_var('float64',init=0);ey=T.alloc_var('float64',init=0)
   ds=T.alloc_var('float64',init=0);de=T.alloc_var('float64',init=0)
   ratio=T.alloc_var('float64',init=0);height=T.alloc_var('float64',init=0)
   volume=T.alloc_var('float64',init=0);union=T.alloc_var('float64',init=0)
   O[i,j]=0
   if i<N[0] and j<N[0] and E[i,j]!=0:
    ox=G[i,0];oy=G[i,1]
    for k in T.serial(4):
     px[k]=G[i,2*k]-ox;py[k]=G[i,2*k+1]-oy
    for edge in T.serial(4):
     ax=G[j,2*edge]-ox;ay=G[j,2*edge+1]-oy
     bx=G[j,2*((edge+1)%4)]-ox;by=G[j,2*((edge+1)%4)+1]-oy
     next_count=0
     for k in T.serial(count):
      sx=px[(k+count-1)%count];sy=py[(k+count-1)%count]
      ex=px[k];ey=py[k]
      ds=(bx-ax)*(sy-ay)-(by-ay)*(sx-ax)
      de=(bx-ax)*(ey-ay)-(by-ay)*(ex-ax)
      if (ds>=0 and de<0) or (ds<0 and de>=0):
       ratio=ds/(ds-de)
       qx[next_count]=sx+ratio*(ex-sx);qy[next_count]=sy+ratio*(ey-sy)
       next_count=next_count+1
      if de>=0:
       qx[next_count]=ex;qy[next_count]=ey;next_count=next_count+1
     count=next_count
     for k in T.serial(count):
      px[k]=qx[k];py[k]=qy[k]
    area=0
    for k in T.serial(count):
     area=area+px[k]*py[(k+1)%count]-py[k]*px[(k+1)%count]
    area=T.abs(area)*T.float64(0.5)
    height=T.max(T.float64(0),T.min((G[i,8]+G[i,9]*T.float64(.5))-(G[j,8]-G[j,9]*T.float64(.5)),(G[j,8]+G[j,9]*T.float64(.5))-(G[i,8]-G[i,9]*T.float64(.5))))
    volume=area*height
    union=G[i,10]*G[i,11]*G[i,9]+G[j,10]*G[j,11]*G[j,9]-volume
    O[i,j]=volume/(union+T.float64(0.00001))
 @T.prim_func
 def suppress(I:T.Tensor((256,256),'float64'),Order:T.Tensor((256,),'int64'),N:T.Tensor((1,),'int32'),Keep:T.Tensor((257,),'int64')):
  with T.Kernel(1,threads=32):
   tx=T.get_thread_binding()
   removed=T.alloc_local((256,),'int32')
   kept=T.alloc_var('int32',init=0)
   chosen=T.alloc_var('int32',init=0)
   if tx==0:
    for k in T.serial(256):removed[k]=0;Keep[k]=-1
    for position in T.serial(N[0]):
     chosen=T.Cast('int32',Order[position])
     if removed[chosen]==0:
      Keep[kept]=T.Cast('int64',chosen);kept=kept+1
      for candidate in T.serial(N[0]):
       if I[chosen,candidate]>T.float64(.1):removed[candidate]=1
    Keep[256]=T.Cast('int64',kept)
 ART.mkdir(parents=True,exist_ok=True);manifest={}
 for name,func in [('overlap',overlap),('suppress',suppress)]:
  kernel=tilelang.compile(func,target={'kind':'cuda','arch':'sm_89'},execution_backend='cython',compile_flags=['-std=c++17','--fmad=false'])
  path=ART/(name+'.so');shutil.copy2(kernel.adapter.libpath,path)
  source=REPORT/'kernelbench'/(name+'_generated.cu');source.parent.mkdir(parents=True,exist_ok=True);source.write_text(kernel.get_kernel_source())
  manifest[name]={'library':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'generated_cuda_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 manifest.update(capacity=256,geometry_dtype='float64',compiler_flags=['-std=c++17','--fmad=false'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tilelang=tilelang.__version__)
 (ART/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(REPORT/'kernelbench/compile_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2))

class GeometryNMS:
 """Persistent same-process ctypes kernels, compatible with unchanged GRAE Torch."""
 def __init__(self):
  import torch
  self.torch=torch;self.calls={'overlap':0,'suppress':0};self.boundary_pairs=0;self.frames=0
  self.libs={}
  for name,nargs in [('overlap',5),('suppress',5)]:
   lib=ctypes.CDLL(str(ART/(name+'.so')));lib.init.restype=ctypes.c_int;lib.get_last_error.restype=ctypes.c_char_p
   if lib.init()!=0:raise RuntimeError(lib.get_last_error())
   lib.call.argtypes=[ctypes.c_void_p]*nargs;lib.call.restype=ctypes.c_int;self.libs[name]=lib
 def launch(self,name,*args):
  rc=self.libs[name].call(*[x.data_ptr() for x in args],self.torch.cuda.current_stream().cuda_stream)
  if rc:raise RuntimeError(self.libs[name].get_last_error())
  self.calls[name]+=1
 def nms(self,boxes,classes,threshold_low=.1,threshold_high=1.,threshold_yaw=.3):
  import numpy as np
  from tracking_gpu_geometry_20261013 import pack
  from mot_3d.utils import iou3d
  if (threshold_low,threshold_high,threshold_yaw)!=(.1,1.,.3):raise ValueError('Only exact native threshold contract supported')
  n=len(boxes)
  if not n:return [],[]
  if n>256:raise ValueError('Measured capacity256 exceeded; compile explicit larger variant')
  if any(min(b.w,b.l,b.h)<=0 for b in boxes):raise ValueError('Nonpositive dimensions rejected at reader boundary')
  geo,order,eligible=pack(boxes,classes)
  if not np.isfinite(geo).all():raise ValueError('Nonfinite geometry')
  g=np.zeros((256,12),np.float64);g[:n]=geo;e=np.zeros((256,256),np.int32);e[:n,:n]=eligible;o=np.zeros(256,np.int64);o[:n]=order
  t=self.torch;G=t.from_numpy(g).cuda();E=t.from_numpy(e).cuda();Order=t.from_numpy(o).cuda();N=t.tensor([n],dtype=t.int32,device='cuda');I=t.empty((256,256),dtype=t.float64,device='cuda');Keep=t.empty(257,dtype=t.int64,device='cuda')
  self.launch('overlap',G,E,N,I)
  matrix=I.cpu().numpy()
  if not np.isfinite(matrix[:n,:n]).all():raise ValueError('Nonfinite IoU')
  boundary=np.argwhere((eligible!=0)&(np.abs(matrix[:n,:n]-.1)<=1e-7))
  if len(boundary):
   for i,j in boundary:matrix[i,j]=iou3d(boxes[i],boxes[j])[1]
   I.copy_(t.from_numpy(matrix))
  if np.any(np.diag(matrix[:n,:n])<=.1):raise ValueError('Original NMS may not terminate: self IoU <= threshold')
  self.boundary_pairs+=len(boundary);self.frames+=1
  self.launch('suppress',I,Order,N,Keep)
  result=Keep.cpu().numpy();indices=result[:int(result[256])].tolist()
  return indices,[classes[i] for i in indices]

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--compile',action='store_true');args=parser.parse_args()
 if args.compile:compile_kernels()
