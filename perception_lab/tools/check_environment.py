import argparse,json,sys,importlib.metadata,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['vision','mmdet3d'],required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
import torch,mmcv
from mmcv.ops import nms
x=torch.tensor([[0.,0.,1.,1.]],device='cuda');dets,indices=nms(x,torch.tensor([.9],device='cuda'),.5)
assert dets.is_cuda and indices.numel()==1
packages=['torch','torchvision','mmcv','mmengine','nuscenes-devkit','numpy']
if a.kind=='vision':
 import ultralytics,mmseg,onnxruntime
 packages+=['ultralytics','mmsegmentation','onnx','onnxruntime-gpu']
 assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
else:
 import mmdet,mmdet3d,spconv.pytorch as spconv
 from mmdet3d.utils import register_all_modules
 register_all_modules()
 packages+=['mmdet','mmdet3d','spconv-cu118']
 layer=spconv.SubMConv3d(4,4,3,padding=1).cuda().eval();x=spconv.SparseConvTensor(torch.ones((2,4),device='cuda'),torch.tensor([[0,1,1,1],[0,2,2,2]],device='cuda',dtype=torch.int32),[4,4,4],1)
 with torch.no_grad(): y=layer(x)
 assert bool(torch.isfinite(y.features).all())
r={'python':sys.version,'executable':sys.executable,'packages':{k:importlib.metadata.version(k) for k in packages},'torch_cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'cuda_ops':'passed','not_model_inference':True}
(out/'environment.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
with (out/'pip_freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
