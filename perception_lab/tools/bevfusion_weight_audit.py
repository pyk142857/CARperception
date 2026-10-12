"""CPU compare packaged ONNX initializers with pinned PyTorch raw/fused state.
Run after generating fused state with NVIDIA's Conv2d+BN fusion (see report).
"""
import argparse,json
from pathlib import Path
import numpy as np,onnx,torch
from onnx import numpy_helper

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--fused-state',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 state=torch.load(a.model_dir/'bevfusion-det.pth',map_location='cpu')['state_dict'];fused=torch.load(a.fused_state,map_location='cpu');rows=[]
 for file in sorted(a.model_dir.glob('*.onnx')):
  m=onnx.load(file)
  for init in m.graph.initializer:
   n=init.name;v=numpy_helper.to_array(init);key=n.removeprefix('model.').removeprefix('parent.')
   if file.name=='camera.vtransform.onnx':key='encoders.camera.vtransform.downsample.'+key
   candidates=[(origin,s[key]) for origin,s in [('raw',state),('fused',fused)] if key in s and tuple(s[key].shape)==v.shape]
   if not candidates and (n.isdigit() or n=='classes_eye') and v.size>4 and v.dtype.kind=='f':
    possible=[]
    for origin,source in [('raw',state),('fused',fused)]:
     for kk,tt in source.items():
      for op,ww in [('identity',tt)]+([('transpose',tt.T)] if tt.ndim==2 else []):
       if tuple(ww.shape)==v.shape:
        err=float(np.max(np.abs(v-ww.detach().numpy())))
        if err<1e-5:possible.append((err,origin+':'+kk+':'+op,ww))
    if possible:
     _,origin,ww=min(possible,key=lambda q:q[0]);candidates=[(origin,ww)];key=origin
   record={'onnx':file.name,'initializer':n,'shape':list(v.shape),'elements':int(v.size),'dtype':str(v.dtype),'pytorch_key':key}
   if candidates:
    results=[]
    for origin,tensor in candidates:
     w=tensor.detach().numpy();error=np.abs(v.astype(np.float64)-w.astype(np.float64));halferror=np.abs(v.astype(np.float64)-w.astype(np.float16).astype(np.float64));results.append({'origin':origin,'max_abs':float(error.max()),'mean_abs':float(error.mean()),'exact_elements':int((v==w).sum()),'after_fp16_rounding_max_abs':float(halferror.max()),'after_fp16_rounding_exact_elements':int((v==w.astype(np.float16)).sum())})
    record['matches']=results;record['best']=min(results,key=lambda r:min(r['max_abs'],r['after_fp16_rounding_max_abs']))
   else:record['unmapped']=True
   rows.append(record)
 summary={}
 for name in sorted({r['onnx'] for r in rows}):
  rr=[r for r in rows if r['onnx']==name];mapped=[r for r in rr if 'best' in r];exact=[r for r in mapped if min(r['best']['max_abs'],r['best']['after_fp16_rounding_max_abs'])==0];summary[name]={'initializers':len(rr),'mapped':len(mapped),'exact_or_fp16_exact':len(exact),'mapped_elements':sum(r['elements'] for r in mapped),'exact_or_fp16_exact_elements':sum(r['elements'] for r in exact),'max_matched_abs':max((min(r['best']['max_abs'],r['best']['after_fp16_rounding_max_abs']) for r in mapped),default=None),'matched_within_1e5':sum(min(r['best']['max_abs'],r['best']['after_fp16_rounding_max_abs'])<1e-5 for r in mapped),'unmapped':[r['initializer'] for r in rr if 'unmapped' in r]}
 a.out.write_text(json.dumps({'summary':summary,'initializers':rows},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
