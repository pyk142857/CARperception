# BEVFusion R50 reference — deployment evidence

This stage uses NVIDIA CUDA-BEVFusion's official `model.zip` ResNet50 checkpoint, six 256×704 cameras and LiDAR with up to nine preceding sweeps. It is a nuScenes mini teaching diagnostic; pretrained training data may overlap mini. It is not full nuScenes validation.

## Source and compatibility

- BEVFusion source: `db75150717a9462cb60241e36ba28d65f6908607`, isolated in `third_party/bevfusion_4090`.
- Config: NVIDIA `configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/{default,convfuser}.yaml`, recursively combined with pinned MIT parent configs. The package `model/resnet50/default.yaml` is only an override, not a standalone configuration.
- Checkpoint: `checkpoints/deployment_4090/nvidia/model/resnet50/bevfusion-det.pth`; source package downloaded by the main deployment stage from the NVIDIA README's NVBox link.
- Environment: isolated `envs/bevfusion`, Python 3.10, read-only dependency path to existing `envs/maptr` Torch 1.13.1+cu117, MMCV 1.7.1, MMDetection 2.28.2. TorchPack 0.3.1 and its additional dependencies installed locally. Full freeze retained.
- CUDA 11.8 compiled all 12 upstream extensions for SM89. Upstream older recommended versions were not assumed compatible: compilation, source import, CPU strict loading and real preprocessing were tested separately.
- Compatibility patch: `tools/bevfusion_torch113_sm89.patch`. Replaces removed, unused THC declarations/includes in six PointNet source files with ATen CUDAContext; explicitly registers original pinned spconv implementations over MMCV's identically named registry entries; compiles SM89. No sparse weight layout modification, tensor reshape or skipped weight layers.
- `strict_cpu.log`: all checkpoint keys matched successfully with `strict=True`.
- `preprocess_check.log`: first real scene-0061 frame generated six images `[6,3,256,704]` and points `[248834,5]`.

## Input and output conventions

Camera order follows the pinned converter: front, front-right, front-left, back, back-left, back-right. Each camera/sweep uses its own sensor calibration and acquisition ego pose transformed to current LIDAR_TOP. Images and points use pinned deterministic test preprocessing. Ground-truth annotation and BEV segmentation target loaders are removed; no label enters forward. `LoadPointsFromMultiSweeps(test_mode=True)` consumes the nine most recent actual sweeps, with upstream close-point removal and empty-history padding. Current-frame point time is zero; sweep time is seconds before the keyframe. Raw intensity is retained.

Native boxes are converted using pinned `output_to_nusc_box`: native dimensions become nuScenes w/l/h without swapping, native yaw becomes `-yaw - pi/2`, and native bottom centers become gravity centers. The resulting native LiDAR boxes, velocities and quaternions are rigidly transformed to current LiDAR ego. All boxes have native→ego→native center, full velocity and rotation roundtrip checks. The standardized contract stores horizontal ego velocity, as in the existing project.

The evaluator reuses `export_records` and its verified ego→global conversion, changing only true sensor metadata to camera+LiDAR. It invokes unmodified official `DetectionEval/detection_cvpr_2019`. Attributes remain empty because this model predicts no attributes; this affects official attribute error and NDS. No rendering threshold is applied to evaluation predictions.

## Reproduction

From `perception_lab`:

```bash
bash tools/bevfusion_setup.sh
envs/bevfusion/bin/python tools/bevfusion_4090.py \
  --source third_party/bevfusion_4090 \
  --model-config third_party/bevfusion_4090/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml \
  --checkpoint checkpoints/deployment_4090/nvidia/model/resnet50/bevfusion-det.pth \
  --dataroot data/nuscenes \
  --packets reports/deployment_4090_20261008/inputs/scene0061_packets.json \
  --out reports/deployment_4090_20261008/bevfusion_reference/scene0061 \
  --capture-pooling runs/deployment_4090_20261008/pooling_captures
# Repeat with mini_val_packets.json and an independent mini_val output directory.
envs/nuscenes_eval/bin/python tools/bevfusion_evaluate.py \
  --predictions reports/deployment_4090_20261008/bevfusion_reference/scene0061/predictions.json \
  --dataroot data/nuscenes --eval-set mini_scene_0061 --scene scene-0061 \
  --out reports/deployment_4090_20261008/bevfusion_reference/scene0061/evaluation
```

Exact matching adapter/config hierarchy/checkpoint/raw sensor/packet hashes and output hashes are required before reusing completed inference artifacts. Partial predictions preserve executed-frame evidence but never imply successful coverage.



## Measured results

Both fixed splits completed with one resident FP32 model, three independent warmups and strict checkpoint loading. Every executed frame has raw/native and standardized outputs; six-camera backbone, LiDAR backbone and fusion hooks all fired. The first smoke exposed missing dataset `box_type_3d` metadata; the adapter now supplies the pinned LiDAR box class and mode, exactly as the upstream dataset pre-pipeline. Failure logs are retained.

| Split | Frames | mAP | NDS | GPU forward P50/P95 ms | End-to-end P50/P95 ms | Peak allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| scene0061 | 39 | 0.711761 | 0.622348 | 122.337/126.767 | 244.279/286.477 | 2.181/2.295 |
| mini_val | 81 | 0.579033 | 0.514471 | 134.695/146.202 | 269.513/401.976 | 2.182/2.863 |

These are desktop RTX4090 measurements. The CUDA-event forward includes upstream head postprocessing and output-to-CPU work; synchronized wall time is also retained. End-to-end includes sensor I/O, deterministic preprocessing, H2D, forward and standardized box conversion, but excludes output-file writing and rendering. CUDNN TF32 was enabled by the installed PyTorch default, while matmul TF32 was disabled; the FP32 label describes tensor dtype, not IEEE-only arithmetic. Capture forwards are separate from measured forwards. PyTorch allocator figures do not represent total process GPU memory.

Scene-0061 used 244,111–255,918 post-filter points; mini_val used 207,998–264,273. Each frame has native→ego→native center, orientation and full 3D velocity roundtrip checks; maximum errors are below 1e-7. Horizontal-only velocity in the standardized contract intentionally discards ego vertical velocity. Exact per-frame timings and input shapes are under each split directory; official class/distance AP and unmodified official metrics are under `evaluation/`.

Real pooling inputs, coordinates, outputs and dimensions from scene frames 0/19/38 were captured outside the measured forward in `runs/deployment_4090_20261008/pooling_captures/`. Each capture is about 656 MiB and remains local. Unified S6 handles representative images, six-camera/BEV visualization and case review. No official full-val or attribute-prediction claim is made.

## Visual evidence and diagnosis

Unified S6 generated six-camera+BEV stills for [frame 0](../visualize/bevfusion_reference/frame_00.jpg), [frame 19](../visualize/bevfusion_reference/frame_19.jpg), and [frame 38](../visualize/bevfusion_reference/frame_38.jpg), plus a 39-frame video with decoded frame-count verification. [Failure-case evidence](scene0061/failure_cases.json) uses a fixed rendering score threshold and same-class 2 m Hungarian assignment for teaching diagnosis. It is separate from official multi-threshold nuScenes AP, and possible causes remain hypotheses.
