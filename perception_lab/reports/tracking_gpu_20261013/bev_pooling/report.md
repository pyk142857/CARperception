# BEV pooling: feature-offset precomputation candidate

Status: **CPU preparation complete; GPU correctness and timing pending**. This preparation does not establish a speedup or justify adoption. Current upstream pooling remains the default. GPU work is deferred until tracking stages B–E have been published and the shared RTX4090 is available.

One candidate is implemented in `perception_lab/tools/bev_pooling_preindex_20261013.cu`: precompute each sorted projection's feature offset once per forward, then use the original pack10 mapping and sequential FP32 FMA accumulation. It does not alter the upstream source, current shared library, frozen detector output, interval order, interval end boundary, camera layout, tensor precision or reduction ordering. No older tile4 variant is combined with it.

## Actual inputs and hotspot hypothesis

Existing corrected first/middle/last captures are at `perception_lab/runs/deployment_4090_20261008/trt_pooling_captures/frame_{0,19,38}`. The `trt_pooling_captures_before_geometry_fix` directory is explicitly excluded. All five binary hashes per capture match the historical corrected `input_statistics.json`. Their native layout is features NHWC FP16, depth NDHW FP16, sorted indices uint32, intervals int32 triples, output NCHW FP16. These are actual FP16 model captures; this candidate does not invent a corresponding FP32 capture or convert the inputs to a different precision.

Each capture has shape `[N,C,D,H,W]=[6,80,118,32,88]`, output `[80,360,360]`, 1,993,728 allocated sorted-index entries and 16,896 unique feature pixels. Input feature/depth/index buffers occupy 2,703,360 / 3,987,456 / 7,974,912 bytes; output is 20,736,000 bytes. The real interval tables occupy approximately 1.16 MB.

| Capture | Intervals | Valid projections | Length P50 / P95 / P99 / maximum | Top 1% intervals' projection share |
|---|---:|---:|---|---:|
| frame_0 | 97,326 | 1,835,478 | 14 / 49 / 107 / 886 | 10.01% |
| frame_19 | 96,046 | 1,834,584 | 14 / 50 / 107 / 859 | 9.92% |
| frame_38 | 96,306 | 1,834,277 | 14 / 49 / 106 / 889 | 9.79% |

The original pack10 kernel launches eight channel threads per interval, each repeating the same camera/pixel division and remainder for every contributing projection. The candidate changes only address preparation: its new kernel generates `(rank / (D*H*W) * H*W + rank % (H*W)) * C`. Original projection order and depth rank remain intact. The main pooling kernel reads that offset and performs the same ten sequential accumulations per thread, with identical 8×128 threads per block.

At source level, the candidate reduces repeated division/remainder pairs from approximately 14.68 million to 1.99 million per frame. The tradeoff is a 7,974,912-byte persistent workspace, an extra kernel launch, approximately 15.95 MB of logical index-read/offset-write traffic, plus about 58.7 MB of logical offset reads by the channel threads. These are logical counts; cache behavior and actual DRAM traffic have not been measured. Long intervals and scattered output writes remain unchanged, so the candidate may lose despite reducing integer arithmetic.

## Correctness and measurement boundary

CPU analysis validated metadata, finite input values, interval range/order/disjointness, unique output cells and all active feature-offset bounds. It also verified complete binary hashes against the corrected historical captures. [input_analysis.json](input_analysis.json) records each result, data volume, configuration/source/library hash and compiler identity.

The independent `.so` was compiled for SM89 using CUDA 11.8 without running GPU work. [compile.log](compile.log) records 16 registers for offset preparation and 40 for pooling, with zero local spills. A compiler result is not a runtime correctness result.

The prepared benchmark will:

- Compare the current actual upstream C ABI (mode 0) against the independent candidate, and verify the upstream output against captured output.
- Require candidate/upstream bitwise equality because FMA order is unchanged. The original fixed FP16 budget (`atol=0.02`, `rtol=0.002`) remains the captured-output acceptance budget; FP32 budget is not applied to these FP16 tensors.
- Check empty/single/collision/boundary fixtures and all three real captures, including their full interval coverage.
- Warm both sides separately, then collect 30 alternating AB/BA pairs per real capture, preserving raw CSV, device-event and synchronized host-wall timings, output hashes and process/memory evidence.
- Include output memset **and fresh index preparation on every candidate forward**, with no caching across frames or amortization outside timing. Both variants receive resident inputs and preallocated outputs; candidate workspace is persistent. H2D/D2H, capture I/O and allocation are excluded equally from this local experiment and recorded as that boundary.
- Classify local benefit only if both order groups improve, at least 24/30 pairs are faster and candidate P95 does not increase, for all three real captures.

If stable local benefit exists, adoption still requires an independent full-model integration, separate 39/81-frame BEVFusion predictions and official detection evaluation, fixed weights/configuration, the existing geometry/intermediate/repeatability checks, and mAP/NDS drops each no greater than 0.005. New predictions must never replace the frozen tracking inputs. If local benefit fails, keep upstream and publish the measured failure; full-model reruns are then unnecessary under this bounded research stage.

No BEVPoolV3 implementation is claimed. This is a single project CUDA candidate grounded in the existing upstream implementation.

## Reproduction

From repository root, CPU preparation only:

```bash
mkdir -p perception_lab/runs/tracking_gpu_20261013/bev_pooling
/usr/local/cuda-11.8/bin/nvcc -O3 -std=c++14 -arch=sm_89 -Xcompiler -fPIC --shared perception_lab/tools/bev_pooling_preindex_20261013.cu -o perception_lab/runs/tracking_gpu_20261013/bev_pooling/libpreindex.so -Xptxas=-v
perception_lab/envs/optipmb_20261013/bin/python perception_lab/tools/bev_pooling_preindex_20261013.py analyze
```

After B–E publication and GPU scheduling, the prepared GPU command is:

```bash
source perception_lab/tools/bevfusion_trt_env.sh
perception_lab/envs/optipmb_20261013/bin/python perception_lab/tools/bev_pooling_preindex_20261013.py benchmark --rounds 30 --warmup 10
```

The benchmark command has **not yet been run** in this preparation report. The `.so`, raw captures and environments remain local; source, hashes, compile command and evidence are shareable.
