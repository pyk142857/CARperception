"""Minimal real FP32 TileLang launch; deliberately not a geometry implementation."""
import argparse, ctypes, hashlib, importlib.util, json, pathlib, shutil
import torch
P=pathlib.Path(__file__).resolve().parent
R=P.parents[2]
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def tensor_sha(x): return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ctypes-only',action='store_true');args=ap.parse_args()
    artifact=R/'runs/tracking_gpu_20261013/tilelang_smoke';artifact.mkdir(parents=True,exist_ok=True)
    libpath=artifact/'fp32_add.so'
    if not args.ctypes_only:
        import tilelang
        import tilelang.language as T
        @T.prim_func
        def add(A:T.Tensor((256,), 'float32'), B:T.Tensor((256,), 'float32'), C:T.Tensor((256,), 'float32')):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(256):
                    C[i]=A[i]+B[i]
        kernel=tilelang.compile(add, target={'kind':'cuda','arch':'sm_89'},execution_backend='cython',compile_flags=['-std=c++17'])
        shutil.copy2(kernel.adapter.libpath,libpath)
        (artifact/'fp32_add.cu').write_text(kernel.get_kernel_source())
    lib=ctypes.CDLL(str(libpath));lib.init.restype=ctypes.c_int;lib.get_last_error.restype=ctypes.c_char_p
    rc=lib.init();assert rc==0,(rc,lib.get_last_error())
    lib.call.argtypes=[ctypes.c_void_p]*4;lib.call.restype=ctypes.c_int
    a=torch.arange(256,dtype=torch.float32,device='cuda');b=a*0.5;c=torch.empty_like(a)
    calls=0
    def launch():
        nonlocal calls
        rc=lib.call(a.data_ptr(),b.data_ptr(),c.data_ptr(),torch.cuda.current_stream().cuda_stream)
        assert rc==0,(rc,lib.get_last_error());calls+=1
    launch();torch.cuda.synchronize();assert torch.equal(c,a+b)
    spec=importlib.util.spec_from_file_location('kb_timing',R/'third_party/KernelBench_20261013/src/kernelbench/timing.py')
    timing=importlib.util.module_from_spec(spec);spec.loader.exec_module(timing)
    ms=timing.time_execution_with_cuda_event(launch,[],num_warmup=2,num_trials=3,discard_first=0,verbose=False)
    out={'mode':'grae_ctypes' if args.ctypes_only else 'tilelang_compile_ctypes','torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'sm':torch.cuda.get_device_capability(),'shape':[256],'dtype':'float32','a_sha256':tensor_sha(a),'b_sha256':tensor_sha(b),'output_sha256':tensor_sha(c),'exact_equal':True,'actual_call_count':calls,'compiled_library':str(libpath),'compiled_library_sha256':sha(libpath),'kernel_source_sha256':sha(artifact/'fp32_add.cu'),'kernelbench_cold_l2_ms':ms}
    name='grae_ctypes_smoke.json' if args.ctypes_only else 'tilelang_smoke.json'
    (P/name).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
