# TASK_STATE — mini 教学闭环已验证

## 当前补充任务：恢复权重与完整评价（2026-10-13，指令已发布）

BOSS授权恢复HARP-NeXt、MambaFusion完整权重并完成整网评价。执行 [20261013_02](instruction/20261013_02_weights_recovery_full_evaluation.md)，基础候选仍遵循20261013_01。
已核验HARP Hub nuScenes32×480文件/revision/大小/SHA256，以及Mamba作者usercontent首尾64KiB Range响应；[来源探测](reports/perception_weights_recovery_20261013/source_probe.json)。**尚未完成完整权重下载、严格加载或新GPU评价。**
执行需包含Mamba官方DetectionEval、HARP原始点语义评价、新Mamba→GRAE和真实回放；保留既有partial及未采用优化结论，不把发布指令写成实验完成。


## 当前候选：2025方法限定（2026-10-13，partial）

BOSS要求“找点2025的，旧的不考虑”。当前候选更新为DEIMv2-S、MambaFusion、GRAE-3DMOT、HARP-NeXt；20261012_02方案已被替代，不继续执行其候选部署。
[当前指令](instruction/20261013_01_2025_perception_deployment.md) · [方案与资源](reports/perception_2025_20261013/plan.md)。
本轮实测：DEIMv2-S COCO val5000 AP0.50861及mini六相机234张；GRAE封存检测输入39/81帧官方跟踪评价（mini_val AMOTA0.82296、IDS13）。HARP/Mamba完整权重缺失，整网与相应预测回放受阻。HARP预处理、Mamba CUDA算子实测不计为完整模型部署。后端、优化采用决定、真实失败案例与回放验收见[执行报告](reports/perception_2025_20261013/report.md)。既有旧模型仅保留历史状态。


2026-09-23，BOSS 明确改为理解流程：仅用 nuScenes mini，现有模型优先，其余模型作为扩展。

## 4090 扩展部署（2026-10-12）

- BEVFusion R50：严格加载 NVIDIA 官方权重，真实六相机与 LiDAR 融合完成 39 + 81 帧，官方 DetectionEval 已评估。
- TensorRT 8.6 FP16：本机五个 engine（含失败的默认 head 与修复用 LayerNorm 插件 head）已构建；两个 split 的 mAP/NDS 下降预算通过。全部低分提案的几何诊断尚有超阈值项，不能宣称数值完全一致。
- BEV pooling：自主可切换 CUDA 候选、真实输入与边界正确性、90 次配对测量和内存检查完成；候选更慢，保持上游默认。Nsight Compute 因驱动性能计数器权限受限，未虚构占用率数据。
- Sparse4D v3：相机输入、原生时序 ID、跨场景重置与序列恢复检查，39 + 81 帧官方检测/跟踪评估。
- GaussianWorld：原分辨率、25,600 Gaussian、FP32 连续 39 帧，全部预测占用网格保存；缺 SurroundOcc 标签，评估为 blocked_data。
- 新结果独立入口 http://127.0.0.1:9094/；统一编排 `tools/run_4090_deployment.py`。完整验收与可复现证据见 [报告](reports/deployment_4090_20261008/report.md)、[状态](reports/deployment_4090_20261008/status.json)。

## 当前完成范围

- M00 数据 / 标定投影；M01 YOLO、M02 SegFormer、M03 Metric Depth：首时刻六路相机真实结果。
- M04 PointPillars：单帧与历史 sweeps；M05 CenterPoint：scene-0061 全部 39 帧。
- M11 作者跟踪器：消费本次 M05 检测，39 帧。
- M12：新增 mini 前视图 PyTorch GPU → ONNX Runtime GPU 数值核对，通过。
- 新增三栏教学回放：原始前视图、CenterPoint 三维框、跟踪 ID / 轨迹，39 帧，2 fps。
- 教学检查核对八个阶段既有产物的 SHA256、逐帧 token、视频完整解码与导出记录。既有推理输出复用，不宣称本轮全部重新推理。

## 入口

- reports/mini_learning.html：主学习入口。
- LEARNING_GUIDE.md：输入 / 输出说明、学习顺序、逐模块复现命令。
- results/mini_learning.json：教学验收证据与 run ID。
- outputs/replay/mini_learning.mp4：同步回放。
- configs/execution_plan.yaml 的 active_scope：当前范围；旧 jobs 仍保留。

## 范围边界

原全量 A/B/C 未达到，但不再是当前目标。全量 nuScenes / Cityscapes / 占用标签不再阻塞本教学闭环。旧规划 M06–M08、M10 与训练微调不作为 mini 闭环前置条件；新增融合、占用和 TensorRT 扩展的实际状态见上方 4090 报告；已有 COCO 评测为附加历史结果。历史闭环不宣称多传感器模型融合；新增 BEVFusion 已实测融合，仍不宣称车端实时性能或全量泛化成绩。

模型运行命令见 LEARNING_GUIDE.md；不要直接运行旧 --phase all，它包含全量评测和扩展。当前无新后台训练或下载任务。

## 本地 Rerun（2026-09-23）

独立环境 envs/rerun，固定 SDK 0.23.4。新增 tools/rerun_mini.py、tools/start_rerun.py 和 RERUN.md。已导出 scene-0061 的 39 帧 / 234 张相机图、点云、GT、检测框及跟踪轨迹；实际 Chrome 渲染通过。服务使用回环地址 9090 / 9091 / 9876，通过 HTTP 加载记录以避开本机 inotify 上限。无模型重推理，无开机自启；可用启动器 --stop 停止。

## MapTR + Rerun（2026-09-24）

独立 envs/maptr（PyTorch 1.13.1 / MMCV 1.7.1）；官方 tiny R50 24e 权重严格加载通过。scene-0061 39 帧真实相机推理，阈值 0.5 下 429 条逐帧矢量预测。Rerun 增加 BEV 面板及三维分隔线/道路边界/人行横道图层，逐帧清理旧线。无地图真值输入；CAN 动态字段缺失采用上游零填充；显示高度为近似地面。结果与摘要：outputs/maptr；代码和复现见 MAPTR.md。

验证：18 项测试通过；234 个投影矩阵与既有官方 infos 对比误差 <1e-9；RRD 3D/BEV 所有帧所有折线逐点对应 JSON，rrd verify 通过；Chrome 实际渲染截图 outputs/rerun/maptr_verified.png。此结果不构成独立精度评估。

右侧六路图像已加入逐帧 MapTR 投影线（image/maptr）；近面和图像边界裁剪测试通过，全套 22 测试通过，六路记录均有有效线段且位于画面内。Chrome 渲染截图 outputs/rerun/camera_lanes_visible.png。使用近似地面，不判断前景遮挡。

## LiDARSeg（2026-09-24）

官方 mini 扩展已补齐，404 帧 / 14,026,208 个标签逐帧点数核对通过。作者 nuScenes Cylinder3D 权重 274 张量严格加载；保留原版 SpConv 1，避免 SpConv 2 缓存变更造成预测错误。scene-0061 完成39帧 / 1,354,112点真实推理，独立环境envs/lidarseg_legacy引用envs/maptr。Rerun加入预测着色、GT页签和BEV语义颜色；原始点文件与结果哈希、所有逐点标签、25测试及Chrome显示通过。复现与边界见LIDARSEG.md；不宣称独立评估或404帧全部推理。

## 跟踪可视化、评估与结果发布（2026-09-28 更新）

已完成 BEV 旋转跟踪框、ID、最多 20 个历史显示位置的轨迹，以及六相机三维框投影；同 ID 跨视图固定配色。当前 39 帧诊断（score≥0.25，中心距离<2m）：检测 TP/FP/FN=2235/1020/75，跟踪=1177/579/39；连续帧 ID 切换 46 次、间隔后换 ID 38 次。报告与 10 张案例图见 [mini_evaluation/report.md](reports/mini_evaluation/report.md)，32 项测试通过，GT 过滤与官方 devkit 一致。该评估不是官方 mAP/NDS/AMOTA。

results/metrics.csv、results/status.json、results/mini_learning.json 与 reports/final_report.md 改为随仓库发布；final_report 是 2026-09-23 历史快照，不代表最新功能状态。MapTR / LiDARSeg 摘要副本位于 reports/published_results/。后续实验按 AGENTS.md 上传结果、报告及 README 入口。

## M05 官方代码 mini 检测评估（2026-09-28）

独立 devkit 1.2.0 / DetectionEval / detection_cvpr_2019，mini_scene_0061 完整 39 帧通过。mAP=0.699071191，NDS=0.610700323，mAAE=1（空属性）；不是完整 val 成绩。报告见 [official_detection_mini/report.md](reports/official_detection_mini/report.md)。复用既有实测 8399 框，未重跑 GPU；35 项测试通过，40 项 AP 官方复核通过，旧诊断与 M11 未改。M05 原 full val evaluation 仍 pending；当前确切状态 blocked_full_val，缺完整 trainval 数据和覆盖全 val 的实测预测。

## Rerun 失败案例（2026-09-28）

固定阈值诊断已加入检测/跟踪独立 BEV 页签、相机检测失败投影与逐帧案例清单；39 帧计数对齐，37 测试通过。原官方 AP 与 M11 结果不变。见 [报告](reports/rerun_failures/report.md)。

## 嵌入式案例浏览（2026-09-28）

新增本地 9092 案例浏览器，固定 Rerun WebViewer 0.23.4，提供筛选、详情、上一条/下一条和自动暂停跳帧，复用已有录制与案例。见 [报告](reports/embedded_case_browser/report.md)。

## 失败事件工作流（2026-09-29）

已实现 mini 离线诊断的目标事件归并、现象分组、人工复核与 Rerun 有界片段播放。1910 条失败→1268 事件→84 分组；不是模型优化结果。线上无真值挖掘和场景级因果归因未实现。见 [报告](reports/failure_events/report.md)。

## 异常默认显示（2026-09-29）

9092 改为独立异常主记录，正常目标按需下载并可隐藏，事件分支联动BEV/3D/相机。正常匹配观测：检测2155、跟踪1067；模型与原指标不变。见 [报告](reports/failure_display/report.md)。

## 置信度统计（2026-09-29）

完成0.10–0.90的10档阈值重评及0.25基线分数分档，原案例/总计完全复现。FN无分数；ID取当帧新轨迹分数。跟踪器未重跑。见 [分析报告](reports/confidence_analysis/analysis-report.md)。

## 4090 部署恢复发布与一致性补查（2026-10-12）

原部署 571 路径恢复并核验，保留原完成提交和最新主分支文档。四帧各后端 5 次生产/5 次捕获运行完成；40 组同输入模块回放及 8 组几何对照完成。发现稀疏深度/LiDAR BEV 变异、几何分配差异和候选交换；head 同源仍有残差，尚未定位具体敏感层。生产模型未改，复用已核验的 39/81 帧评估，原全候选失败保留。Pooling 无稳定提速、默认 upstream；Sparse4D 证据复核，GaussianWorld 推理完成但 GT 缺失。22 项测试通过。详见[补查报告](reports/deployment_4090_20261012_followup/report.md)和[远端回执](reports/deployment_4090_20261012_followup/publication_receipt.json)。
