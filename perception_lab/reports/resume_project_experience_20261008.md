# CARperception 简历项目经历与成果依据

整理日期：2026-10-08。面向自动驾驶感知算法实习、三维感知与算法工程岗位。

以下成稿围绕仓库已完成的模型部署、评估诊断、失败案例工具、跟踪关联与 CPU 优化。对应工程与实验快照为 [main 提交 cfe4083](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/README.md)（2026-09-30）；项目起止时间和个人职责按本人实际情况填写。

## 按招聘平台模板填写

### 项目名称

自动驾驶感知算法部署与跟踪优化（CARperception）

### 担任角色

感知算法开发与工程实现

### 项目时间

开始时间：2026.09  
结束时间：[填写实际结束月份；仍在开展则选择“至今”]

仓库中的部署与实验记录始于 2026 年 9 月，个人参与起止时间按实际填写。

### 项目描述

基于 nuScenes mini 搭建多相机与激光雷达感知实验平台，使用 Python、PyTorch、MMDetection3D 部署 CenterPoint 三维检测、PubTracker 多目标跟踪、Cylinder3D 点云分割及 MapTR 矢量地图预测。实现标定与坐标转换、Rerun 多视图同步回放，接入 nuScenes 官方评估及失败案例复核；完成 YOLOv8s ONNX 导出与 GPU 输出数值核验，并开展跟踪关联和 CPU 性能优化。

### 项目业绩

- 完成 39 帧、234 幅相机图像及约 135 万个激光雷达点的离线处理和同步回放，形成模型输出、诊断报告与复现记录。
- 固定检测输入完成 13 组跟踪对照，在 mini 单场景中将连续 ID 切换由 46 降至 38 次，减少 17.4%；行人 ID 切换由 42 降至 36 次，误检与漏检数量不变。
- 通过 NumPy 批量计算、复制快路径与代价矩阵复用，将候选 CPU 跟踪函数平均耗时从 2.91 降至 0.77 ms/帧，加速 3.77 倍；完成 30 轮配对计时，39 帧输出逐项一致，该阶段通过 72 项测试。
- 构建可视化复核工作台，将 1,910 条失败记录归并为 1,268 个事件、84 个现象分组，支持置信度/距离筛选、目标定位、片段回放和人工标签修正。

### 项目链接

https://github.com/pyk142857/CARperception

量化跟踪结果限定于 nuScenes mini 的 scene-0061；CPU 提速限定于候选跟踪函数。项目采用预训练模型，完整官方 val 与跨场景泛化验证尚未完成。

## 已实现任务与范围

| 任务 | 模型或实现 | 已完成范围 |
|---|---|---|
| 数据与几何适配 | FramePacket、标定、位姿与投影 | 多相机和点云元数据整理；LiDAR/自车/世界/相机变换；三维框、速度与轨迹适配；六相机投影核对 |
| 二维检测 | YOLOv8s | 首时刻六路相机结果；另有 COCO val 5,000 图评估；耗时剖析独立补测前三时刻共 18 图 |
| 图像语义分割 | SegFormer | 首时刻六路相机推理及标签数组；耗时剖析独立补测 18 图 |
| 米制深度 | Depth Anything V2 Metric Depth | 首时刻六路相机预测及稀疏 LiDAR 投影诊断；耗时剖析独立补测 18 图 |
| 三维检测 | PointPillars、CenterPoint | PointPillars 单帧与历史 sweeps 推理；CenterPoint 连续 39 帧输出及官方代码检测评估 |
| 多目标跟踪 | PubTracker、实验候选跟踪器 | 39 帧七类目标跟踪；max_age 与 13 组关联配置对照；输出、关联诊断与身份事件归档 |
| 点云语义分割 | Cylinder3D | 39 帧、1,354,112 点预测；16 个语义类及 ignore；mini 的 404 帧标签完成点数审计 |
| 矢量地图预测 | MapTR tiny R50 | 39 帧六相机推理；分隔线、道路边界、人行横道三类折线；显示阈值 0.5 下 429 条逐帧预测 |
| 导出核验 | YOLOv8s ONNX、ONNX Runtime GPU | 真实单图、FP32、opset 17 输出核验；核对 CUDA 执行，保留输出数组与误差记录 |
| 可视化与排障工具 | Rerun、Python 服务、JavaScript 工作台 | BEV、三维、六相机与统一时间轴；异常定位、筛选、高亮、特写、片段回放、逐条人工标签及历史导出 |
| 评估与性能分析 | 官方 devkit、诊断评估、Profiler | 四距离 AP/PR；固定阈值 FP/FN/身份事件；置信度和距离分析；CPU 配对计时及 GPU 同步阶段计时 |

相机三分支的 18 图补测属于性能分析的独立工作量；原教学回放中的三分支结果仍以首时刻为范围。PointPillars 与 CenterPoint 是两种检测方案；多个结果在同一界面展示，不等于已实现 Camera-LiDAR 融合模型。

## 量化成果与直接证据

| 成果 | 数据或协议 | 可核对结果 | 证据 |
|---|---|---|---|
| 跟踪配置对照 | 固定检测、39 帧、score≥0.25、同类中心距离<2 m；13 组含基线与镜像校验 | 全类连续切换 46→38，减少 17.4%；行人 42→36，减少 14.3%；FP 579、FN 39 不变 | [实验报告](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/tracking_optimization/report.md)、[原始指标](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/tracking_optimization/metrics.csv) |
| 身份错误合计 | 连续切换与中断后换 ID 按原诊断分别记录 | 84→75，减少 10.7%；候选未替换默认回放 | [实验报告与定义](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/tracking_optimization/report.md) |
| CPU 跟踪提速 | i9-14900KF；预热后 30 轮交替顺序；每种实现 1,170 个计时样本 | 2.910→0.772 ms/帧；3.77×；延迟下降 73.5%；P95 4.219→1.022 ms/帧 | [性能报告](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/tracking_operator_optimization/report.md)、[原始摘要](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/tracking_operator_optimization/summary.json) |
| 优化正确性 | 39 帧、4,547 个跟踪输入框 | 全部轨迹字段、顺序、ID、诊断精确一致，输入未被修改；该阶段 72 项测试通过 | [测试日志](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/tracking_operator_optimization/tests.txt)、[优化实现](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/tools/fast_candidate_tracker.py) |
| 失败事件组织 | 同一基线的检测与跟踪逐帧失败；非独立目标计数 | 1,910 条记录→1,268 个事件→84 个现象分组；每条有效记录恰好归属一个事件 | [事件报告](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/failure_events/report.md) |
| 官方检测评估接入 | devkit 1.2.0，mini 自定义单场景 split，0.5/1/2/4 m | mAP 0.6991、NDS 0.6107；40 项 AP 官方函数复核；空属性使 mAAE=1 | [官方评估报告](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/official_detection_mini/report.md)、[原始指标](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/official_detection_mini/metrics/metrics_summary.json) |
| 点云分割部署 | scene-0061，忽略真值 0；1,016,188 个有效评估点 | 单场景准确率 95.16%；非零并集类别平均 IoU 71.99%；未作为独立测试成绩 | [部署说明](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/LIDARSEG.md)、[原始摘要](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/published_results/lidarseg_summary.json) |
| 标定与 MapTR 展示 | 39 帧六相机；官方权重严格加载 | 234 个投影矩阵与官方 infos 对比误差<1e-9；429 条逐帧预测，非独立道路元素数 | [任务状态](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/TASK_STATE.md)、[MapTR 实现说明](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/MAPTR.md) |
| 推理耗时诊断 | 13 个离线任务；模型计时边界显式 GPU 同步 | 总任务 123.48 s；导入/元数据/权重加载占 53.2%，绘图/视频占 13.9%；常驻 CenterPoint 探针约 25.22 ms/次 | [完整计时口径](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/pipeline_timing/report.md) |
| ONNX 数值核验 | 真实 mini 前视图；PyTorch GPU 对 ONNX Runtime GPU；FP32 | mini 导出阶段验收通过；脚本记录 CUDA provider、数值误差及输出数组 | [教学验收](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/results/mini_learning.json)、[导出代码](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/tools/export_yolo.py) |

简历主稿优先采用跟踪身份稳定性和 CPU 提速两类配对结果。检测与分割高分数来自 mini 场景，可能与预训练数据重叠，适合说明评估管线已运行，不用于宣称独立泛化性能。失败记录归并减少的是重复浏览量，未改变模型精度。

## 面试可展开的实际工作

### 坐标与上游模型适配

说明为什么检测框和速度需先转到世界坐标做跨帧关联，再转回当前自车坐标显示；轨迹历史也保存在世界坐标以避免自车运动污染。三维框导出需区分重心与底面中心、w/l/h 顺序、四元数和速度旋转。相机投影考虑标定与位姿，当前工作台没有动态目标曝光时刻外推和遮挡判断。

Cylinder3D 的旧权重依赖 SpConv 1 的核布局和索引缓存语义。最终保留原算子和严格加载，逐点核对预处理与输出顺序；这个排障案例可展示模型部署与兼容性诊断能力。

### 跟踪策略如何选择

高分框先关联，低分框再续接已有轨迹；新建轨迹阈值单独控制。实验同时看连续切换、间隔后换 ID、FP/FN 与近前方漏检。提高新建阈值到 0.5 虽能大幅减少 ID 切换，却增加漏检；最终候选保留 0.25 新建阈值。匈牙利和卡尔曼在本场景的退化结果也保存，以实际对照选择方案。

### 性能优化如何证明正确

复制快路径仍隔离可变列表/数组；批量计算保留 float32 转换位置和顺序；贪心匹配保留相同代价下的选择顺序。先比对全部输出与输入未修改，再预热并交替测量参考/优化版本。3.77× 的计时排除模型推理、文件加载、坐标变换和渲染，因此只表述为 CPU 跟踪函数提速。

## 成果范围

- 核心感知网络使用开源预训练模型，本项目成果覆盖适配、部署、评估、诊断工具及跟踪策略/CPU 实现优化。
- 完整 nuScenes val、跨场景候选验证、官方 AMOTA、模型训练/微调、TensorRT 和量化尚未完成。
- BEVFusion、BEVFormer、CenterFusion、SurroundOcc 尚未完成；MapTR 已完成多视角相机 BEV 地图预测。
- 当前是离线实验与本机复核工作台；未完成车端部署、规划控制闭环或统一实时流水线。
- 2 fps 为视频回放速率；CenterPoint 常驻探针为首帧预热后重复调用，尚不能推算生产端到端 FPS。
- 各阶段测试数代表当时执行的测试集，不把多个报告的测试数量相加。

## 更多实现入口

[数据和任务状态](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/TASK_STATE.md) · [原始检测/跟踪诊断](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/mini_evaluation/report.md) · [人工标签功能](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/manual_case_labels/report.md) · [距离置信度分析](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/distance_confidence_cross/report.md) · [完整点云与取景](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/reports/full_pointcloud/report.md) · [教学回放视频](https://github.com/pyk142857/CARperception/blob/cfe4083664fa1de377d6a84156bdcd42d5defffd/perception_lab/outputs/replay/mini_learning.mp4)

