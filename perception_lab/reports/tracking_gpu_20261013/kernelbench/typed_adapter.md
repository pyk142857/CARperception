# KernelBench project task extension

This adapter targets the pinned KernelBench commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1`. It is a project task extension, not a standard 250-task benchmark score. Upstream remains unchanged.

`typed_adapter_conceptual.diff` shows the explicit replacement of upstream `_process_input_tensor`'s blanket `input.to(dtype=precision)` by a device-only transfer. Packed corners/height/dimensions and IoU remain float64; eligible-pair matrix and count remain int32; sorting and keep indices remain int64. This also avoids upstream's TileLang fp16/bfloat16-only assertion. `compare_output` first requires shape and dtype equality, uses exact equality for integers and ordered indices/types, and checks every finite FP64 element against `1e-8 + 1e-7*abs(reference)` (FP32 alternative: `1e-5 + 1e-5*abs(reference)`).

CPU adapter tests exercise FP64 low bits, int32 values above exact FP32 integer range, and int64 values above exact FP64 integer range. All three demonstrably lose information under the original FP32 cast. Rejection tests cover integer off-by-one/order, dtype mismatch, excessive FP64 error and nonfinite outputs. CPU test results are in `typed_cpu_validation.json`.

GPU mode independently reconstructs real first/middle/last frames from each of three scenes (nine frames), prepares typed inputs, executes actual compiled TileLang kernels, compares eligible-pair IoUs with original SimpleTrack Shapely/height semantics, compares ordered indices/types exactly, and verifies the integrated NMS path including its boundary rechecks. It records input/output shape, dtype and hash, source provenance and actual launch counts. Root's 120-frame correctness report remains the full coverage evidence.

GPU mode imports and calls unmodified KernelBench `timing.py::time_execution_with_cuda_event` directly. Its L2-flushing device-resident overlap+suppress timings are explicitly labeled cold-L2 diagnostics. They exclude CPU packing, transfers and boundary repair, and are not mixed with the separate required warm-cache AB/BA full-path benchmark. Timing helper source hash is recorded. There is no LLM API or cloud GPU dependency.

Run from repository root, after compiling the existing geometry libraries:

```bash
CUDA_VISIBLE_DEVICES='' perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/tracking_gpu_kernelbench_20261013.py --cpu-only
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/tracking_gpu_kernelbench_20261013.py --gpu --trials 3
```

GPU execution writes `typed_gpu_validation.json` only after successful completion. Absence of that file means the GPU adapter stage has not completed; CPU tests alone do not establish kernel correctness or performance.
