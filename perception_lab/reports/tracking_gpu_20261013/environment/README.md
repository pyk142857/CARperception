# TileLang / KernelBench isolated environment

Sources are pinned in `source_manifest.json`. TileLang 0.1.15 is installed from its wheel; all 390 matching Python source files are byte-identical to commit a35f8ddf45eba16c21211ec8822d56ce5363036f. Compiler library and patched header hashes are recorded. KernelBench source is an unchanged checkout of 423217d9fda91e0c2d67e4a43bf62f96f6d104f1. No external LLM services or cloud GPU were used.

The new venv inherits the existing autolabel Python 3.10 / Torch 2.1.2+cu118 site-packages read-only; package installation uses `--no-deps`. Original GRAE and BEVFusion environments are unchanged. New environment and source checkouts consume approximately 343 MB total, without another Torch/CUDA runtime wheel download.

## Reproduce

From repository root:

```bash
/home/minglei/anaconda3/envs/autolabel/bin/python -m venv --system-site-packages perception_lab/envs/tilelang_20261013
perception_lab/envs/tilelang_20261013/bin/python -m pip install --no-deps -r perception_lab/reports/tracking_gpu_20261013/environment/requirements_local.txt
patch -p0 -d perception_lab/envs/tilelang_20261013/lib/python3.10/site-packages < perception_lab/reports/tracking_gpu_20261013/environment/cuda12_2_header_compat.patch
CUDA_HOME=/usr/local/cuda-12.2 PATH=/usr/local/cuda-12.2/bin:$PATH TVM_FFI_DISABLE_TORCH_C_DLPACK=1 perception_lab/envs/tilelang_20261013/bin/python perception_lab/reports/tracking_gpu_20261013/environment/tilelang_smoke.py
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/reports/tracking_gpu_20261013/environment/tilelang_smoke.py --ctypes-only
```

Source checkout commands:

```bash
git clone --filter=blob:none --no-checkout https://github.com/tile-ai/tilelang.git perception_lab/third_party/tilelang_0_1_15
git -C perception_lab/third_party/tilelang_0_1_15 checkout a35f8ddf45eba16c21211ec8822d56ce5363036f
git clone --filter=blob:none --no-checkout https://github.com/ScalingIntelligence/KernelBench.git perception_lab/third_party/KernelBench_20261013
git -C perception_lab/third_party/KernelBench_20261013 checkout 423217d9fda91e0c2d67e4a43bf62f96f6d104f1
```

## Compatibility findings and integration

- Current TVM requires target attributes as `{'kind':'cuda','arch':'sm_89'}`; older CLI target strings are rejected.
- CUDA 12.2 with GCC 9 cannot use upstream default C++20. Explicit `compile_flags=['-std=c++17']` overrides it. The unused `st_bulk_shared` template also contains unconditional `static_assert(false)` rejected before instantiation by this compiler. The isolated one-line patch makes the condition dependent on the template argument but still always false, preserving rejection if the unsupported operation is instantiated. FP32 add source contains no bulk-shared operation. Original source checkout stays unchanged.
- Optional Torch C-DLPack allocator initialization warns on Torch 2.1; compilation and raw pointer execution still work. GRAE ctypes integration does not import TileLang or tvm_ffi.
- Use `tilelang.compile(...,execution_backend='cython', compile_flags=['-std=c++17'])`, copy `kernel.adapter.libpath`, save `kernel.get_kernel_source()`. The library exposes `init()`, `call(tensor pointers...,cudaStream_t)`, `get_last_error()`; explicit `ctypes.c_void_p` signatures prevent pointer truncation. Check return codes and synchronize to detect asynchronous failures. GRAE can allocate old-Torch CUDA tensors and pass `data_ptr()` plus current stream. Compilation occurs outside steady-state timing; actual launches, transfers and synchronization remain inside the relevant measured path.
- `T.alloc_local((16,2),'float64')` gives thread-private clipping scratch; `T.alloc_var('int32',init=...)`, `T.get_thread_binding()`, `T.serial(n)`, and dynamic control flow are available. A one-pair-per-thread FP64 clipping implementation can preserve original corner coordinates and original height formula. This is an API feasibility finding; the smoke is not geometry correctness evidence.

## KernelBench usage boundary

Unmodified `eval.py::_process_input_tensor` casts every tensor to a single precision and `eval_kernel_against_ref` rejects TileLang FP32/FP64. Do not use these directly for mixed class/order/geometry tensors. Project adaptation must preserve per-input dtypes, compare integer NMS outputs exactly and apply project FP64/FP32 tolerances separately.

`timing.py::time_execution_with_cuda_event` is imported directly without loading LLM/dataset packages and is exercised by the smoke. It flushes L2 before every trial; the saved three timings are only smoke diagnostics, not the task's required warm-cache AB/BA performance result. For final paired timing, retain warm cache and include CPU sorting/coarse filtering/H2D/D2H in their proper layers.
