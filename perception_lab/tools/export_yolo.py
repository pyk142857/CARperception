"""Export official YOLOv8s and compare decoded outputs on one real image."""
import argparse,json,shutil,subprocess,sys,os
os.environ['NVIDIA_TF32_OVERRIDE']='0'
from pathlib import Path
import numpy as np
import torch
from evidence import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--export-only',action='store_true');p.add_argument('--image');args=p.parse_args()
 torch.set_num_threads(2)
 from ultralytics import YOLO
 from ultralytics.data.augment import LetterBox
 import cv2,onnx,onnxruntime as ort
 root=Path(__file__).resolve().parents[1];out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 weight=root/'checkpoints/yolov8s.pt'
 if args.export_only:
  shutil.copy2(weight,out/'yolov8s.pt')
  model=YOLO(out/'yolov8s.pt')
  model.export(format='onnx',imgsz=640,batch=1,dynamic=False,opset=17,simplify=False,device='cpu')
  return
 # Ultralytics select_device('cpu') hides CUDA process-wide; isolate export.
 subprocess.run([sys.executable,__file__,'--out',str(out),'--export-only'],check=True)
 filename=out/'yolov8s.onnx'
 graph=onnx.load(filename);onnx.checker.check_model(graph)
 image_path=Path(args.image) if args.image else root/'data/bus.jpg'
 image=cv2.imread(str(image_path))
 if image is None:raise ValueError('Cannot read image: '+str(image_path))
 x=LetterBox((640,640),auto=False,stride=32)(image=image)
 x=np.ascontiguousarray(x[:,:,::-1].transpose(2,0,1)[None]).astype(np.float32)/255
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 native=YOLO(weight).model.float().eval().fuse().cuda()
 with torch.inference_mode():a=native(torch.from_numpy(x).cuda())[0].cpu().numpy()
 options=ort.SessionOptions();options.intra_op_num_threads=2;options.enable_profiling=True;options.profile_file_prefix=str(out/'ort_profile')
 session=ort.InferenceSession(str(filename),sess_options=options,providers=[('CUDAExecutionProvider',{'gpu_mem_limit':1073741824,'cudnn_conv_algo_search':'HEURISTIC','cudnn_conv_use_max_workspace':'0'})])
 session.disable_fallback()
 if 'CUDAExecutionProvider' not in session.get_providers():raise RuntimeError('CUDA EP absent; reject CPU fallback')
 b=session.run(None,{session.get_inputs()[0].name:x})[0]
 profile=Path(session.end_profiling());events=json.loads(profile.read_text())
 providers={e.get('args',{}).get('provider') for e in events if e.get('cat')=='Node'};providers.discard(None)
 if 'CUDAExecutionProvider' not in providers:raise RuntimeError('No recorded CUDA execution')
 if not np.isfinite(b).all():raise RuntimeError('ONNX produced NaN/Inf')
 error=np.abs(a-b);relative=error/np.maximum(np.abs(a),1e-6)
 result={'scope':'single_real_image_numerical_diagnostic','source':'measured','image':str(image_path.resolve()),'image_sha256':sha256(image_path),'checkpoint_sha256':sha256(weight),'onnx_sha256':sha256(filename),
 'shape':list(x.shape),'output_shape':list(b.shape),'opset':17,'dtype':'float32','providers':session.get_providers(),
 'profile_providers':sorted(providers),'profile':str(profile),'absolute_error_max':float(error.max()),'absolute_error_mean':float(error.mean()),
 'relative_error_max':float(relative.max()),'relative_error_mean':float(relative.mean()),'finite':True,
 'precision_policy':'FP32; NVIDIA_TF32_OVERRIDE=0; PyTorch matmul/cudnn TF32 disabled',
 'full_task_regression':'pending','tensorrt':'pending','performance':'not measured under isolated GPU load'}
 np.savez_compressed(out/'numerical_outputs.npz',pytorch=a,onnxruntime_gpu=b)
 (out/'numerical_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 # Small FP32 differences are checked independently of full-dataset AP tolerance.
 np.testing.assert_allclose(b,a,rtol=1e-3,atol=1e-3)
if __name__=='__main__':main()
