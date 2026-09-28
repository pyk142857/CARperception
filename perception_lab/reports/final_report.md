# Perception lab — partial

> 历史报告快照：生成于 2026-09-23，2026-09-28 上传。以下指标与原始阶段记录保留；后续已完成 MapTR、LiDARSeg、Rerun 多视图及检测跟踪诊断，请参阅 [当前状态](../TASK_STATE.md) 和 [最新评估](mini_evaluation/report.md)。原全量 A/B/C 仍未达到。

A/B/C 均未达到。已通过项目仅代表该阶段及记录的数据范围。
示例图片和 mini 工程诊断不构成正式 val 回放。没有生成完整联合回放视频，也没有完成训练或部署收益对比。

## 实测指标

| 模块 | 指标 | 值 | 范围 | 样本数 |
|---|---|---:|---|---:|
| M01 | AP | 0.449536 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AP50 | 0.617743 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AP75 | 0.486248 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AP_small | 0.260055 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AP_medium | 0.499275 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AP_large | 0.610304 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AR1 | 0.356251 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AR10 | 0.591276 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AR100 | 0.644612 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AR_small | 0.438812 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AR_medium | 0.709423 | official pycocotools bbox COCO 2017 val | 5000 |
| M01 | AR_large | 0.800143 | official pycocotools bbox COCO 2017 val | 5000 |
| M03 | AbsRel | 0.460359 | nuScenes single-sweep LiDAR camera-Z sparse depth diagnostic; 0.1<Z<=80m; nearest pixel Z; no scale alignment | 1 |
| M03 | RMSE | 8.299268 | nuScenes single-sweep LiDAR camera-Z sparse depth diagnostic; 0.1<Z<=80m; nearest pixel Z; no scale alignment | 1 |
| M03 | delta1 | 0.287418 | nuScenes single-sweep LiDAR camera-Z sparse depth diagnostic; 0.1<Z<=80m; nearest pixel Z; no scale alignment | 1 |

## 阶段状态

| 模块 | 阶段 | 状态 | 原因 |
|---|---|---|---|
| M00 | smoke | passed |  |
| M00 | replay | pending | Official trainval two-scene replay manifest and raw assets absent; mini is not a substitute |
| M00 | visualization | pending | Not yet executed; see TASK_STATE.md and data/resource requirements |
| M00 | preflight | passed |  |
| M01 | assets | passed |  |
| M01 | environment | passed |  |
| M01 | smoke | passed |  |
| M01 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M01 | evaluation | passed |  |
| M01 | visualization | blocked | Mini diagnostic figures exist under smoke/history_smoke; required full-val visualizations await trainval and replay |
| M01 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M01 | export | blocked | Not yet executed; see TASK_STATE.md and data/resource requirements; No validated execution command configured |
| M01 | export_validation | blocked | Not yet executed; see TASK_STATE.md and data/resource requirements; No validated execution command configured |
| M01 | diagnostic | passed |  |
| M02 | assets | passed |  |
| M02 | environment | passed |  |
| M02 | smoke | passed |  |
| M02 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M02 | evaluation | pending | Cityscapes official val images and fine labels absent; official download requires login |
| M02 | visualization | blocked | Mini diagnostic figures exist under smoke/history_smoke; required full-val visualizations await trainval and replay |
| M02 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M02 | export | blocked | Not yet executed; see TASK_STATE.md and data/resource requirements; No validated execution command configured |
| M02 | export_validation | blocked | Not yet executed; see TASK_STATE.md and data/resource requirements; No validated execution command configured |
| M02 | diagnostic | passed |  |
| M03 | assets | passed |  |
| M03 | environment | passed |  |
| M03 | smoke | passed |  |
| M03 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M03 | evaluation | pending | Full official nuScenes val assets unavailable |
| M03 | visualization | blocked | Mini diagnostic figures exist under smoke/history_smoke; required full-val visualizations await trainval and replay |
| M03 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M03 | export | blocked | Not yet executed; see TASK_STATE.md and data/resource requirements; No validated execution command configured |
| M03 | export_validation | blocked | Not yet executed; see TASK_STATE.md and data/resource requirements; No validated execution command configured |
| M03 | diagnostic | passed |  |
| M03 | sparse_depth_diagnostic | passed |  |
| M04 | assets | passed |  |
| M04 | environment | passed |  |
| M04 | smoke | passed |  |
| M04 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M04 | evaluation | pending | Full official nuScenes val assets unavailable |
| M04 | visualization | blocked | Mini diagnostic figures exist under smoke/history_smoke; required full-val visualizations await trainval and replay |
| M04 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M04 | export | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M04 | export_validation | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M04 | history_smoke | passed |  |
| M05 | assets | passed |  |
| M05 | environment | passed |  |
| M05 | smoke | passed |  |
| M05 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M05 | evaluation | pending | Full official nuScenes val assets unavailable |
| M05 | visualization | blocked | Mini diagnostic figures exist under smoke/history_smoke; required full-val visualizations await trainval and replay |
| M05 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M05 | export | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M05 | export_validation | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M05 | history_smoke | passed |  |
| M05 | mini_scene | passed |  |
| M06 | assets | passed |  |
| M06 | environment | pending | Legacy environment not implemented/verified; current modern mmcv2 environment must not be substituted |
| M06 | smoke | pending | Native adapter and legacy environment not yet integrated; see reports/compatibility.md. No model inference claimed. |
| M06 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M06 | evaluation | pending | Full official nuScenes val assets unavailable |
| M06 | visualization | pending | No measured predictions for this module |
| M06 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M06 | export | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M06 | export_validation | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M07 | assets | passed |  |
| M07 | environment | pending | Legacy environment not implemented/verified; current modern mmcv2 environment must not be substituted |
| M07 | smoke | pending | Native adapter and legacy environment not yet integrated; see reports/compatibility.md. No model inference claimed. |
| M07 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M07 | evaluation | pending | Full official nuScenes val assets unavailable |
| M07 | visualization | pending | No measured predictions for this module |
| M07 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M07 | export | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M07 | export_validation | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M08 | assets | blocked | Official centerfusion_e60 checkpoint link returned HTTP404; logs/legacy_asset_probe.json; No validated execution command configured |
| M08 | environment | pending | Legacy environment not implemented/verified; current modern mmcv2 environment must not be substituted |
| M08 | smoke | pending | Native adapter and legacy environment not yet integrated; see reports/compatibility.md. No model inference claimed. |
| M08 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M08 | evaluation | pending | Full official nuScenes val assets unavailable |
| M08 | visualization | pending | No measured predictions for this module |
| M08 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M08 | export | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M08 | export_validation | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M09 | assets | passed |  |
| M09 | environment | pending | Legacy environment not implemented/verified; current modern mmcv2 environment must not be substituted |
| M09 | smoke | pending | Native adapter and legacy environment not yet integrated; see reports/compatibility.md. No model inference claimed. |
| M09 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M09 | evaluation | pending | Full official nuScenes val assets unavailable |
| M09 | visualization | pending | No measured predictions for this module |
| M09 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M09 | export | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M09 | export_validation | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M10 | assets | blocked | SurroundOcc semantic checkpoint and matched labels not obtained from author Baidu links; No validated execution command configured |
| M10 | environment | pending | Legacy environment not implemented/verified; current modern mmcv2 environment must not be substituted |
| M10 | smoke | pending | Native adapter and legacy environment not yet integrated; see reports/compatibility.md. No model inference claimed. |
| M10 | replay | pending | Official nuScenes trainval and frozen two-scene val manifest absent |
| M10 | evaluation | pending | Full official nuScenes val assets unavailable; SurroundOcc matched occupancy labels also missing |
| M10 | visualization | pending | No measured predictions for this module |
| M10 | benchmark | blocked | No controlled GPU measurement window; existing user GPU jobs remain active; No validated execution command configured |
| M10 | export | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M10 | export_validation | not_applicable | Full 3D TensorRT is optional in this plan; native GPU and benchmark remain required |
| M11 | replay_M05 | pending | Full official val CenterPoint detections absent |
| M11 | replay_M07 | pending | BEVFusion measured detections absent |
| M11 | evaluation_M05 | pending | Official val tracking inputs absent |
| M11 | evaluation_M07 | pending | Official val tracking inputs absent |
| M11 | benchmark | blocked | Full required tracking runs not ready; No validated execution command configured |
| M11 | mini_tracking_M05 | passed |  |
| M12 | onnx_diagnostic_M01 | passed |  |
| M12 | onnx_gpu_M01 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M12 | tensorrt_fp16_M01 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M12 | export_validation_M01 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M12 | onnx_gpu_M02 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M12 | tensorrt_fp16_M02 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M12 | export_validation_M02 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M12 | onnx_gpu_M03 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M12 | tensorrt_fp16_M03 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M12 | export_validation_M03 | blocked | Complete backend integration/full-dataset regression not yet implemented or run; TensorRT engine not built. See diagnostic substeps separately.; No validated execution command configured |
| M13 | train_A | blocked | 80 official train scenes and full val baseline unavailable; no training or optimizer updates performed; No validated execution command configured |
| M13 | train_B | blocked | 80 official train scenes and full val baseline unavailable; no training or optimizer updates performed; No validated execution command configured |
| M13 | evaluation | blocked | 80 official train scenes and full val baseline unavailable; no training or optimizer updates performed; No validated execution command configured |
| M14 | diagnostic_cases | passed |  |
| M14 | unified_replay | pending | Required official two-val-scene predictions across all modules absent; mini model-specific videos are separate diagnostics |

## 证据与限制

- [完整状态](../results/status.json)；[实测指标](../results/metrics.csv)；[缺失资产](missing_assets.md)；[资源预算](resource_budget.md)。
- [仓库 commit](../configs/locked/repositories.json)；[权重来源和哈希](../checkpoints/manifest.json)。
- GPU 存在其他用户任务；尚无满足50次预热、200次测量的独占GPU性能结果。performance.csv只有表头，不填入伪FPS。
- 未完成 Cityscapes、nuScenes trainval、SurroundOcc 官方评估；训练A/B均未执行，不宣称精度提升。
- 已保存 8 个本次COCO预测的阈值定义漏检诊断案例。它们不替代官方COCO AP，也不证明天气/遮挡等成因；未知属性明确为unknown。
- 新建隔离环境复用已有PyTorch，依赖已有基础环境路径；版本冻结见envs/。
- M04–M10 TensorRT可行性不等同导出；自定义稀疏卷积、BEV pooling、deformable attention仍需逐目标验证。
- 重跑入口及已验证命令见[README](../README.md)。

- [M05 / mini_scene mini诊断视频](../runs/20260923T043110_d6ae8066/M05/mini_scene/mini_centerpoint_bev.mp4)
- [M11 / mini_tracking_M05 mini诊断视频](../runs/20260923T044014_0542fd21/M11/mini_tracking_M05/mini_tracking_M05.mp4)
### M00 / smoke
![M00 smoke](../runs/20260923T042215_98dd4e4f/M00/smoke/six_camera_projection.jpg)

### M00 / smoke
![M00 smoke](../runs/20260923T042215_98dd4e4f/M00/smoke/bev_GT.png)

### M01 / smoke
![M01 smoke](../runs/20260923T042332_c329422c/M01/smoke/000004_overlay.jpg)

### M01 / smoke
![M01 smoke](../runs/20260923T042332_c329422c/M01/smoke/000002_overlay.jpg)

### M01 / smoke
![M01 smoke](../runs/20260923T042332_c329422c/M01/smoke/000000_overlay.jpg)

### M01 / smoke
![M01 smoke](../runs/20260923T042332_c329422c/M01/smoke/000003_overlay.jpg)

### M01 / smoke
![M01 smoke](../runs/20260923T042332_c329422c/M01/smoke/000005_overlay.jpg)

### M01 / smoke
![M01 smoke](../runs/20260923T042332_c329422c/M01/smoke/000001_overlay.jpg)

### M01 / diagnostic
![M01 diagnostic](../runs/20260923T040056_be065e33/M01/diagnostic/000000_overlay.jpg)

### M02 / smoke
![M02 smoke](../runs/20260923T042335_518032ba/M02/smoke/000003_overlay.png)

### M02 / smoke
![M02 smoke](../runs/20260923T042335_518032ba/M02/smoke/000002_overlay.png)

### M02 / smoke
![M02 smoke](../runs/20260923T042335_518032ba/M02/smoke/000001_overlay.png)

### M02 / smoke
![M02 smoke](../runs/20260923T042335_518032ba/M02/smoke/000004_overlay.png)

### M02 / smoke
![M02 smoke](../runs/20260923T042335_518032ba/M02/smoke/000005_overlay.png)

### M02 / smoke
![M02 smoke](../runs/20260923T042335_518032ba/M02/smoke/000000_overlay.png)

### M02 / diagnostic
![M02 diagnostic](../runs/20260923T040334_ab91aa3c/M02/diagnostic/000000_overlay.png)

### M03 / smoke
![M03 smoke](../runs/20260923T042412_1fa019a6/M03/smoke/000002_depth.png)

### M03 / smoke
![M03 smoke](../runs/20260923T042412_1fa019a6/M03/smoke/000000_depth.png)

### M03 / smoke
![M03 smoke](../runs/20260923T042412_1fa019a6/M03/smoke/000001_depth.png)

### M03 / smoke
![M03 smoke](../runs/20260923T042412_1fa019a6/M03/smoke/000003_depth.png)

### M03 / smoke
![M03 smoke](../runs/20260923T042412_1fa019a6/M03/smoke/000004_depth.png)

### M03 / smoke
![M03 smoke](../runs/20260923T042412_1fa019a6/M03/smoke/000005_depth.png)

### M03 / diagnostic
![M03 diagnostic](../runs/20260923T040144_a0b3e125/M03/diagnostic/000000_depth.png)

### M03 / sparse_depth_diagnostic
![M03 sparse_depth_diagnostic](../runs/20260923T042604_ebca3122/M03/sparse_depth_diagnostic/CAM_BACK_RIGHT_error.png)

### M03 / sparse_depth_diagnostic
![M03 sparse_depth_diagnostic](../runs/20260923T042604_ebca3122/M03/sparse_depth_diagnostic/CAM_FRONT_LEFT_error.png)

### M03 / sparse_depth_diagnostic
![M03 sparse_depth_diagnostic](../runs/20260923T042604_ebca3122/M03/sparse_depth_diagnostic/CAM_FRONT_RIGHT_error.png)

### M03 / sparse_depth_diagnostic
![M03 sparse_depth_diagnostic](../runs/20260923T042604_ebca3122/M03/sparse_depth_diagnostic/CAM_BACK_error.png)

### M03 / sparse_depth_diagnostic
![M03 sparse_depth_diagnostic](../runs/20260923T042604_ebca3122/M03/sparse_depth_diagnostic/CAM_BACK_LEFT_error.png)

### M03 / sparse_depth_diagnostic
![M03 sparse_depth_diagnostic](../runs/20260923T042604_ebca3122/M03/sparse_depth_diagnostic/CAM_FRONT_error.png)

### M04 / smoke
![M04 smoke](../runs/20260923T042445_9f92e472/M04/smoke/bev_predictions.png)

### M04 / history_smoke
![M04 history_smoke](../runs/20260923T042914_b9db5872/M04/history_smoke/bev_predictions.png)

### M05 / smoke
![M05 smoke](../runs/20260923T042552_955c73a9/M05/smoke/bev_predictions.png)

### M05 / history_smoke
![M05 history_smoke](../runs/20260923T042924_4d9aae03/M05/history_smoke/bev_predictions.png)

### M11 / mini_tracking_M05
![M11 mini_tracking_M05](../runs/20260923T044014_0542fd21/M11/mini_tracking_M05/000038_tracks.png)

### M11 / mini_tracking_M05
![M11 mini_tracking_M05](../runs/20260923T044014_0542fd21/M11/mini_tracking_M05/000033_tracks.png)

### M11 / mini_tracking_M05
![M11 mini_tracking_M05](../runs/20260923T044014_0542fd21/M11/mini_tracking_M05/000005_tracks.png)

### M11 / mini_tracking_M05
![M11 mini_tracking_M05](../runs/20260923T044014_0542fd21/M11/mini_tracking_M05/000019_tracks.png)

### M11 / mini_tracking_M05
![M11 mini_tracking_M05](../runs/20260923T044014_0542fd21/M11/mini_tracking_M05/000035_tracks.png)

### M11 / mini_tracking_M05
![M11 mini_tracking_M05](../runs/20260923T044014_0542fd21/M11/mini_tracking_M05/000024_tracks.png)

### M14 / diagnostic_cases
![M14 diagnostic_cases](../runs/20260923T042605_390a93c7/M14/diagnostic_cases/000000003501.jpg)

### M14 / diagnostic_cases
![M14 diagnostic_cases](../runs/20260923T042605_390a93c7/M14/diagnostic_cases/000000002685.jpg)

### M14 / diagnostic_cases
![M14 diagnostic_cases](../runs/20260923T042605_390a93c7/M14/diagnostic_cases/000000002431.jpg)

### M14 / diagnostic_cases
![M14 diagnostic_cases](../runs/20260923T042605_390a93c7/M14/diagnostic_cases/000000001675.jpg)

### M14 / diagnostic_cases
![M14 diagnostic_cases](../runs/20260923T042605_390a93c7/M14/diagnostic_cases/000000002149.jpg)

### M14 / diagnostic_cases
![M14 diagnostic_cases](../runs/20260923T042605_390a93c7/M14/diagnostic_cases/000000000776.jpg)
