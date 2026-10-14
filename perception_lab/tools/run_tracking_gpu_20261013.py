"""Small phase dispatcher for the frozen tracking experiment; GPU phases serial."""
import argparse,os,subprocess,sys
from pathlib import Path
LAB=Path(__file__).resolve().parents[1]
ROOT=LAB.parent

def command(env,tool,*args,extra_env=None):
 executable=LAB/'envs'/env/'bin/python'
 run_env=os.environ.copy();run_env.update(extra_env or {})
 subprocess.run([str(executable),str(LAB/'tools'/tool),*args],cwd=ROOT,env=run_env,check=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['inputs','native_compare','profile','kernel','benchmark','full_eval','bev_pooling','report']);a=p.parse_args()
 grae='grae3dmot_2025'
 if a.phase=='inputs':command(grae,'tracking_gpu_inputs_20261013.py')
 elif a.phase=='native_compare':
  command(grae,'grae3dmot_2025.py','--phase','run','--input','frozen_reference','--config',str(LAB/'configs/tracking_gpu_20261013.json'),'--output-dir','reports/tracking_gpu_20261013/native_compare/grae')
  command('optipmb_20261013','optipmb_tracking_20261013.py','--validate')
  command('optipmb_20261013','optipmb_tracking_20261013.py','--diagnostic')
  for model in ['grae','optipmb']:command('sparse4d','tracking_gpu_evaluate_20261013.py','--model',model)
 elif a.phase=='profile':
  command(grae,'tracking_gpu_geometry_20261013.py');command(grae,'tracking_gpu_profile_segments_20261013.py')
 elif a.phase=='kernel':
  command('tilelang_20261013','tracking_gpu_kernel_20261013.py','--compile',extra_env={'CUDA_HOME':'/usr/local/cuda-12.2','PATH':'/usr/local/cuda-12.2/bin:'+os.environ.get('PATH',''),'TVM_FFI_DISABLE_TORCH_C_DLPACK':'1'})
  command(grae,'tracking_gpu_validate_20261013.py');command(grae,'tracking_gpu_boundary_checks_20261013.py');command(grae,'tracking_gpu_kernelbench_20261013.py','--gpu','--trials','3')
 elif a.phase=='benchmark':command(grae,'tracking_gpu_benchmark_20261013.py','--phase','all')
 elif a.phase=='full_eval':
  command(grae,'tracking_gpu_grae_20261013.py');command('sparse4d','tracking_gpu_evaluate_20261013.py','--model','grae_gpu')
 elif a.phase=='report':subprocess.run([sys.executable,str(LAB/'tools/tracking_gpu_report_20261013.py')],cwd=ROOT,check=True)
 elif a.phase=='bev_pooling':
  # Existing upstream TensorRT/CUDA library search paths are required; no rebuild of historical upstream.
  library=LAB/'runs/tracking_gpu_20261013/bev_pooling/libpreindex.so';library.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['/usr/local/cuda-11.8/bin/nvcc','-O3','-std=c++14','-arch=sm_89','-Xcompiler','-fPIC','--shared',str(LAB/'tools/bev_pooling_preindex_20261013.cu'),'-o',str(library),'-Xptxas=-v'],check=True)
  for stage in ['analyze','benchmark']:
   subprocess.run(['bash','-c','source "$1"; exec "$2" "$3" "$4"','pooling',str(LAB/'tools/bevfusion_trt_env.sh'),str(LAB/'envs/tensorrt/bin/python'),str(LAB/'tools/bev_pooling_preindex_20261013.py'),stage],cwd=ROOT,check=True)
if __name__=='__main__':main()
