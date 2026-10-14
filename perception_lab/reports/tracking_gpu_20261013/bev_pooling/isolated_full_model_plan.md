# Conditional isolated full-model integration

**Prepared only. No isolated tree materialized, no model build, no model run and no detection evaluation executed by this preparation.** Use this plan only if the local preindex experiment first demonstrates stable benefit. Old source, old binaries, weights, TensorRT engines and frozen tracking predictions remain untouched.

## Accurate integration points

- `tools/bevfusion_trt_4090.py:35`: selects the extension/core build directory. The prepared runner copy points this to the isolated build, while retaining the exact deployment config/model directory and CPU pipeline.
- `tools/bevfusion_trt_4090.py:76`: calls `libpybev.load_bevfusion` on the existing camera, transform, fusion and bbox plans plus LiDAR ONNX. No engine rebuild is needed.
- `tools/bevfusion_trt_4090.py:97`: actual `core.forward` returns native detections. Output/coordinate/actual-input evidence and original export routines remain intact.
- `tools/bevfusion_trt_4090.py:142`: original paired helper call. The copied runner expands measurement to each scene's first/middle/last actual packet, with 30 AB/BA pairs per representative frame (nine frames across both splits).
- `CUDA-BEVFusion/src/bevfusion/camera-bevpool.cu:210`: the sole `car_bev_pool_forward` call in `BEVPoolImplement::forward`. The prepared patch adds `CAR_BEV_POOLING=preindex` as an independent branch to the new function and preserves existing upstream dispatch.
- `CUDA-BEVFusion/src/python.cpp:269`: `dlopen("libcustom_layernorm.so", RTLD_NOW)`. Reuse that original unchanged plugin through a symlink in the isolated build; retain original plans.
- `CUDA-BEVFusion/CMakeLists.txt`: `CORE_FILES` glob includes `src/bevfusion/*.cu`; copying the new CUDA file there automatically compiles it into the isolated core library.

## Prepared files and scope

`tools/bev_pooling_preindex_isolate_20261013.py` defaults to generating reviewable diffs and an integration manifest. It currently produced:

- [camera-bevpool.cu.patch](camera-bevpool.cu.patch): class-owned persistent uint32 offset buffer, allocation/free lifecycle, and a preindex branch. Buffer capacity is `N*D*H*W`. Every forward recomputes offsets; no cross-frame calibration/geometry caching occurs.
- [bevfusion_preindex_run.py.patch](bevfusion_preindex_run.py.patch): isolated build path, runtime verification of actual extension/core mappings, strict new report subtree, only upstream/preindex modes, nine representative full-model benchmarks.
- [bev_pooling_preindex_full_model.py.patch](bev_pooling_preindex_full_model.py.patch): existing paired helper's candidate mode routes to `preindex`. Its `candidate_*` CSV column names remain the same. This copy is imported for `paired_forwards`; its old standalone summary `main()` is not the evaluator for this experiment.
- [isolated_integration_plan.json](isolated_integration_plan.json): original binary/source and planned modified-file hashes.

`--materialize` copies only the current corrected `src/` plus `CMakeLists.txt` into `runs/tracking_gpu_20261013/bev_pooling/isolated/CUDA-BEVFusion`, creates read-only-use dependency symlinks and writes modified copies. It refuses an existing isolated destination. No full third-party repo, dataset, model weights or engines are duplicated. It does not invoke CMake, nvcc or CUDA. The base is the current corrected project source, not a fresh upstream checkout that would accidentally omit prior geometry/capacity fixes.

The prepared copied runner checks `libpybev.__file__` and `/proc/self/maps` to reject loading the old core by mistake. In the isolated build, both upstream and preindex share exactly the same rebuilt core and model, so compiler changes cannot become a hidden difference between pair members. Before the main experiment, compare an isolated-upstream smoke frame with the existing core's upstream output to detect rebuild drift; preserve any discrepancy rather than attributing it to pooling.

## Conditional commands

These commands are prepared and **not executed**. Run from repository root after GPU availability and local-benefit confirmation:

```bash
python perception_lab/tools/bev_pooling_preindex_isolate_20261013.py --materialize
source perception_lab/tools/bevfusion_trt_env.sh
CAR_PREINDEX_SRC="$PWD/perception_lab/runs/tracking_gpu_20261013/bev_pooling/isolated/CUDA-BEVFusion"
CAR_PREINDEX_BUILD="$CAR_PREINDEX_SRC/build"
cmake -S "$CAR_PREINDEX_SRC" -B "$CAR_PREINDEX_BUILD" \
  -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DProtobuf_INCLUDE_DIR="$BEV_TRT_PROTOBUF/include" \
  -DProtobuf_LIBRARY=/lib/x86_64-linux-gnu/libprotobuf.so.17 \
  -DProtobuf_PROTOC_EXECUTABLE="$BEV_TRT_PROTOBUF/bin/protoc"
cmake --build "$CAR_PREINDEX_BUILD" --target pybev -j4
ln -s "$BEV_TRT_BUILD/libcustom_layernorm.so" "$CAR_PREINDEX_BUILD/libcustom_layernorm.so"
export LD_LIBRARY_PATH="$CAR_PREINDEX_BUILD:$LD_LIBRARY_PATH"
export PYTHONPATH="$CAR_PREINDEX_BUILD:${PYTHONPATH:-}"
```

Use the exact existing `perception_lab/configs/deployment_4090.yaml`. For each `scene0061` and `mini_val`, run both `CAR_BEV_POOLING=upstream` and `CAR_BEV_POOLING=preindex` with the copied runner, `--packets perception_lab/reports/deployment_4090_20261008/inputs/<split>_packets.json`, and:

```text
--out perception_lab/reports/tracking_gpu_20261013/bev_pooling/full_model/<mode>/<split>
--capture-pooling perception_lab/runs/tracking_gpu_20261013/bev_pooling/full_model_captures/<mode>/<split>
```

The executable is the existing BF-capable environment, `perception_lab/envs/bevfusion/bin/python`, and the script is `$CAR_PREINDEX_SRC/tooling/bevfusion_preindex_run.py`. First use `--limit 1` in a distinct smoke directory, then run complete splits without `--limit`. Pass `--paired-pooling-rounds 30` for each preindex split run; do not duplicate those measurements in the upstream run. Both modes are warmed separately inside every paired benchmark. Capture extra forwards happen outside per-frame timings. The helper disables capture while timing pairs.

The inference report root is enforced by code, so accidental output to old reports/frozen detection locations raises an error. New upstream full-model predictions are separate contemporaneous comparison data, not replacements for historical frozen BEVFusion input.

## Evaluation and decision

Use `envs/sparse4d/bin/python tools/bevfusion_evaluate.py` with `--predictions <new-mode-split>/predictions.json`, `--dataroot perception_lab/data/nuscenes`, and `--out <new-mode-split>/official`. `mini_val` uses `--eval-set mini_val`; scene0061 uses a custom split, for example `--eval-set mini_scene_0061 --scene scene-0061`, after checking the existing split definition for agreement. This reuses unmodified official DetectionEval/config_factory(`detection_cvpr_2019`). Keep full input/prediction/config/GT-scope/evaluation hashes.

Before metric comparison, verify config, packet, checkpoint, ONNX and every engine hash, actual preprocessed input hashes, and identical isolated core/extension hashes across upstream/preindex. Keep the candidate CUDA source and isolated source-tree/build hashes separately. Compare fresh captures' depth/features/indices/intervals exactly; pooled output should be bitwise equal because this candidate preserves FMA ordering. Run repeated fixed-input forwards to distinguish model nondeterminism from candidate errors, and use existing geometry budgets for center/size/yaw/velocity/score. Reuse `bev_pooling_full_model.verify_inference_and_evaluation` and `bevfusion_compare.compare` for sealed artifacts and diagnostic matching; do not directly run the old hardcoded full-model summary, whose expected pooling value is `candidate` and old output layout differs.

Report both mAP and NDS changes and each class. Maximum permitted absolute drops are 0.005 each, but passing metric budgets does not excuse tensor, geometry or repeatability failures. Full-model timings include resident model forward/H2D/native box D2H; input preprocessing and geometry-update times are separately reported, as in the original runner. Summarize AB/BA, raw 30 pairs per representative frame, mean/P50/P95 and process competition. Do not infer full-model benefit from the local pooling kernel alone. If any required test fails or full-model benefit is unstable, retain upstream and preserve evidence.
