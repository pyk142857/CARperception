# CARperception 简历项目经历与成果依据

更新日期：2026-10-14。面向自动驾驶感知算法、三维感知与算法工程实习。证据快照：[ca6458c](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/README.md)；早期成稿保存在[2026-10-08简历版本](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/resume_project_experience_20261008.md)，对应实验和本次修订由Git历史追溯。

## 按招聘平台模板填写

### 项目名称

自动驾驶多模态感知部署与跟踪关联算子优化（CARperception）

### 担任角色

感知算法开发与算子优化

个人职责按实际参与情况填写。

### 项目时间

2026.09—至今

### 项目描述

基于nuScenes mini搭建多相机与激光雷达感知实验平台，使用Python、PyTorch、MMDetection3D部署DEIMv2二维检测、BEVFusion融合检测、GRAE-3DMOT跟踪、Cylinder3D点云分割及MapTR矢量地图预测。开发批量速度回推、类别门控、二维中心距离矩阵与置信度两阶段贪心关联算子，接入官方检测/跟踪评价和Rerun六相机、BEV、三维同步回放，形成推理、评估、失败复核、策略改进与性能验证闭环。

### 项目业绩

- 使用TileLang实现FP64旋转框三维IoU与GPU贪心NMS，保持原排序、粗筛、阈值和高度公式；在RTX4090上对固定BEVFusion检测→GRAE的39/81帧完整序列各做30轮AB/BA，合并耗时降低87.09%（约7.75倍），120帧轨迹逐字节一致、官方跟踪指标完全一致。计时从检测JSON读取到tracks，包含搬运，排除BEVFusion前向。
- 完成DEIMv2-S原生部署及COCO val2017全5,000图评价，AP由历史YOLOv8s的44.95%提高至50.86%，提升5.91个百分点；完成nuScenes 39帧六相机234图预测展示。
- 固定CenterPoint检测完成13组跟踪策略对照，在nuScenes mini单场景39帧中将连续帧ID切换46→38，减少17.4%；行人42→36，FP579/FN39保持不变。
- 开发并优化CPU跟踪关联算子，通过复制快路径、NumPy批量计算、类别广播及代价矩阵复用，将同一候选跟踪函数平均耗时2.91→0.77 ms/帧，加速3.77倍；30轮配对计时，39帧全部输出字段、ID、顺序及诊断精确一致，阶段72项测试通过。
- 打通BEVFusion→GRAE-3DMOT融合检测与跟踪链路，完成39帧及81帧mini_val官方代码评价；81帧检测mAP57.90%、跟踪AMOTA82.30%，保存预测、输入哈希、场景重置检查及失败案例。
- 构建多传感器同步复核工作台，将1,910条基线失败记录组织为1,268个事件、84个现象分组，支持距离/置信度筛选、目标定位、六相机与三维联动及片段回放。

### 项目链接

https://github.com/pyk142857/CARperception

简历中的策略收益限定于scene-0061；3.77倍限定于CPU跟踪函数。核心网络使用开源预训练权重。COCO与nuScenes分开报告；nuScenes mini可能与预训练数据重叠，不表述为官方全量val、车端部署或整条感知流水线实时性能。

## 精简版项目业绩

- 部署DEIMv2、BEVFusion、GRAE-3DMOT及点云分割/地图预测，搭建官方评价、失败诊断与六相机/BEV/3D同步复核闭环；COCO 5,000图AP44.95%→50.86%。
- 在固定检测的mini 39帧对照中，两阶段贪心关联将连续ID切换46→38，减少17.4%，FP/FN不变。
- 优化CPU类别门控、二维距离矩阵及关联数据路径，平均耗时2.91→0.77ms/帧、3.77倍提速；30轮配对、完整输出一致。

## TensorRT方向的可选补充表述

若投递部署/推理优化岗位，可在主稿中补充或替换一条：

> 完成DEIMv2-S的ONNX FP32与TensorRT 10.6 FP16导出、构建和完整COCO评价；在单RTX4090、batch1、三张暖缓存图的30轮配对中，TensorRT端到端平均16.66→7.53ms，约2.21倍，COCO AP下降0.0541个百分点；分析并保留FP16置信度跨阈值变化，作为可选后端，默认保持原生。

这是限定范围内的后端部署与性能实测。严格候选一致性尚未通过，不写“已完成默认后端替换”“所有输出完全一致”或“全场景实时加速”。ONNX更慢，不作为加速成果。

## 已保留的方案及最新完成范围

| 任务 | 已验证实现 | 范围及状态 |
|---|---|---|
| 数据与几何适配 | 标定、位姿、LiDAR/自车/世界/相机坐标与框/速度转换 | 多传感器同步、预测与GT分离、投影核对 |
| 优先二维检测 | DEIMv2-S native FP32；YOLOv8s作历史对照 | COCO5,000图官方评价；mini39帧234图展示，无mini二维AP |
| 融合三维检测 | BEVFusion R50 native FP32 | 39/81帧官方DetectionEval，保持上游pooling |
| 学习型三维跟踪 | BEVFusion固定检测→GRAE-3DMOT原生关联 | 39/81帧官方TrackingEval；不将改变检测输入的收益归因于跟踪器 |
| 策略与CPU算子优化 | CenterPoint→S4_greedy_low→FastCandidateTracker | 39帧固定检测对照；仅此链路支撑17.4%及3.77倍 |
| 点云分割 | Cylinder3D | 39帧1,354,112原始点；1,016,188有效点；16类mIoU71.99% |
| 矢量地图预测 | MapTR tiny R50 | 39帧六相机、三类折线；展示而非独立地图精度评价 |
| 导出与GPU部署核验 | DEIM ONNX/TRT及历史YOLO ONNX | 保留真实后端运行、完整精度评价、配对计时和数值诊断 |
| 同步排障 | Rerun与失败案例工作台 | 原始点云、六相机、三维框/轨迹、GT和模型独立图层；39/81帧新回放已验收 |

上述链路分别运行，不能将其排列成已验证的统一实时流水线；特别是FastCandidateTracker尚未在BEVFusion输入上完成同口径验证。两阶段跟踪通过的是单场景候选准入，历史默认回放保留作者基线。

## 成果依据

| 简历表述 | 原始数值与协议 | 直接证据 |
|---|---|---|
| 二维检测AP提升5.91个百分点 | YOLO0.4495358206→DEIM0.5086100690；同一COCO val2017 5,000图、官方COCOeval；不同模型/预训练权重的方案对比 | [YOLO指标](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/results/metrics.csv)、[DEIM官方指标](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/perception_2025_20261013/deimv2/official_eval/metrics.json) |
| 连续ID切换减少17.4% | 固定CenterPoint，39帧，score≥0.25、同类XY<2m；46→38；FP579/FN39；不是官方IDS/AMOTA | [报告](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/tracking_optimization/report.md)、[原始指标](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/tracking_optimization/metrics.csv) |
| CPU跟踪3.77倍 | i9-14900KF，2.910221986→0.772018150ms/帧；30轮交替顺序，双方各1,170样本；不含模型/坐标/IO/渲染 | [配对报告](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/tracking_operator_optimization/report.md)、[摘要](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/tracking_operator_optimization/summary.json)、[实现](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/tools/fast_candidate_tracker.py) |
| 融合检测与跟踪已评价 | 81帧BEVFusion mAP0.5790331434；其固定检测→GRAE AMOTA0.8229635568、IDS13、FP299、FN400 | [检测](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/deployment_4090_20261008/bevfusion_reference/report.md)、[跟踪](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/perception_2025_20261013/grae3dmot/report.md)、[检测输入来源](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/perception_2025_20261013/grae3dmot/frozen_reference/input_manifest.json) |
| 失败事件工作台 | 1,910条逐帧失败→1,268事件→84分组；属于组织与复核功能，未改变模型精度 | [事件报告](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/failure_events/report.md) |
| 可选TensorRT约2.21倍 | native16.664721→TRT7.525682ms；三张图、30轮、batch1暖缓存；COCO5,000图AP0.5086100690→0.5080690341 | [配对原始数据](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/perception_2025_20261013/deimv2/tensorrt/paired_benchmark.json)、[精度及一致性限制](https://github.com/pyk142857/CARperception/blob/ca6458cdb3d5784f9e9bc30cb86272fafe9ebd9a/perception_lab/reports/perception_2025_20261013/deimv2/report.md) |

## 面试可展开的实现细节

- **跟踪算子开发**：类别门控、二维中心距离矩阵、批量速度回推与置信度两阶段贪心匹配均是实际实现；CPU版以NumPy广播替代双层Python比较，复用代价矩阵、减少临时数组和递归复制，并保持相等代价的选择顺序。
- **几何适配**：区分框重心/底面中心、w/l/h、四元数、速度旋转与ego→global变换；历史轨迹在世界坐标保存再投影，跨场景重置。标定投影不等于动态目标曝光时刻外推或遮挡建模。
- **推理部署**：完整权重严格加载、来源与哈希绑定、模块Hook验证实际执行；使用原始评价预测，显示阈值不裁官方AP输入。CPU提速、GPU forward、端到端耗时分别定义。
- **CUDA候选的取舍**：已开发并校验BEV pooling CUDA候选，但没有稳定提速，保留上游；可以讲正确性和性能诊断，不将其写成成功GPU加速。CPU关联算子也不改称CUDA算子。

## 已完成实验与成功优化的区别

MambaFusion已完成39/81帧原生检测及接入GRAE：mini_val检测mAP58.90%、跟踪AMOTA84.97%，但重复一致性未通过，IDS13→19、FP299→405；只在技术交流中作为带明确限制的模型对照，不列入主稿成功优化。HARP-NeXt完成404帧原始点预测和评价，但同39帧mIoU64.42%低于Cylinder3D的71.99%；其GPU预处理候选未采用。GRAE批量D2H导出有顺序依赖、BEVFusion FP16尚有几何一致性问题，都保留实验记录。

完整nuScenes val、模型训练/微调、车端部署、规划控制闭环及统一实时吞吐尚未完成。2fps是回放速率。各阶段测试数只代表相应阶段，不相加为总项目测试量。

[项目保留方案说明](../RETAINED_ALGORITHMS.md) · [最新权重恢复与评价报告](perception_weights_recovery_20261013/report.md) · [2025模型总报告](perception_2025_20261013/report.md)

## 2026-10-14 GPU几何补充证据

[完整实验](tracking_gpu_20261013/report.md) · [30轮全链路原始数据](tracking_gpu_20261013/benchmark/full.json) · [120帧NMS与输入等价](tracking_gpu_20261013/kernel_correctness.json) · [官方整体／每类指标等价](tracking_gpu_20261013/grae_gpu/official_equivalence.json)。

该收益来自CPU多边形几何与抑制的GPU替换，不是BEVFusion网络加速，也不改变GRAE网络／关联数学；不表述为完整感知系统端到端FPS、全量数据泛化或车端量产结果。KernelBench仅用于本项目typed任务适配，不宣称标准250题成绩。
