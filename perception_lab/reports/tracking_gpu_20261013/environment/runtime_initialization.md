# GRAE runtime integration

Compiled TileLang kernels load in the unchanged Torch 1.13.1 GRAE process through ctypes; no per-frame subprocess or IPC. Initialize `torch.cuda.init()` before importing the legacy geometry/Numba dependency graph. Importing that graph before lazy CUDA initialization caused a process segmentation fault; initializing CUDA first ran all 120 frames successfully. This is a process initialization compatibility workaround, not a geometry/math change.

`kernelbench/executed_compile_adapter.py` archives the exact compilation-time source identified by `compile_manifest.json`. The subsequent host adapter adds explicit CUDA initialization; the compiled functions are unchanged. Shared libraries remain local; generated CUDA and hashes are published.
