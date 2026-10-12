"""CPU-only checkpoint Conv/BN folding for the ONNX correspondence audit."""
import argparse,sys,torch
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(root/'third_party/bevfusion_4090'))
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
from mmdet3d.models import build_model
from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn
configs.load(str(root/'third_party/bevfusion_4090/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml'),recursive=True)
cfg=Config(recursive_eval(configs));cfg.model.encoders.camera.backbone.init_cfg=None
model=build_model(cfg.model);state=torch.load(root/'checkpoints/deployment_4090/nvidia/model/resnet50/bevfusion-det.pth',map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True);model.eval()
# NVIDIA ptq.py's exact recursive fusion for ordinary Conv2d.
def fuse(module):
 last=None;name_last=None
 for name,child in module.named_children():
  if isinstance(child,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.SyncBatchNorm)):
   if last is not None:
    if isinstance(last,torch.nn.Conv1d):
     factor=child.weight/torch.sqrt(child.running_var+child.eps)
     bias=last.bias if last.bias is not None else torch.zeros_like(child.running_mean)
     last.weight=torch.nn.Parameter(last.weight*factor[:,None,None]);last.bias=torch.nn.Parameter((bias-child.running_mean)*factor+child.bias)
    else:module._modules[name_last]=_fuse_conv_bn(last,child)
    module._modules[name]=torch.nn.Identity();last=None
  elif isinstance(child,(torch.nn.Conv2d,torch.nn.Conv1d)):last=child;name_last=name
  else:fuse(child)
fuse(model)
result=model.state_dict()
# Exported sparse-conv is channel-last input/output layout then output-major.
mods=list(model.encoders.lidar.backbone.named_modules());index=0
for i,(name,conv) in enumerate(mods):
 if hasattr(conv,'weight') and conv.weight.ndim==5:
  bn=next(m for _,m in mods[i+1:] if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
  factor=bn.weight*torch.rsqrt(bn.running_var+bn.eps)
  weight=conv.weight*factor.reshape(1,1,1,1,-1)
  bias=(torch.zeros_like(bn.running_mean)-bn.running_mean)*torch.rsqrt(bn.running_var+bn.eps)*bn.weight+bn.bias
  result[f'spconv{index}.weight']=weight.permute(4,0,1,2,3).contiguous();result[f'spconv{index}.bias']=bias
  index+=1
torch.save(result,args.out)
