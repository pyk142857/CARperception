# CARperception 简历项目经历与成果依据

更新日期：2026-10-14。面向自动驾驶感知算法、三维感知、推理部署及GPU算子优化实习。证据快照：[73efc89](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/README.md)；文件沿用原路径，历史成稿由Git记录追溯。

## 按招聘平台模板填写

### 项目名称

自动驾驶多模态感知部署与CPU/GPU算子优化（CARperception）

### 担任角色

感知算法开发与部署优化

个人职责按实际参与情况填写。

### 项目时间

2026.09—至今

### 项目描述

基于nuScenes mini搭建六相机与激光雷达感知实验平台，使用Python、PyTorch、MMDetection3D部署DEIMv2二维检测、BEVFusion融合三维检测、GRAE-3DMOT跟踪、Cylinder3D点云分割及MapTR矢量地图预测。接入官方评价与Rerun六相机、BEV、三维同步回放，围绕失败案例开展关联策略、CPU数据路径、GPU几何算子与推理后端优化，形成推理、评估、失败定位、优化、复测闭环。

### 项目业绩

- **GPU算子开发与完整跟踪流程提速：** 使用TileLang实现FP64批量旋转框3D IoU与GPU贪心NMS，保留原类别、排序、粗筛和高度公式；在RTX4090上，将固定BEVFusion检测→GRAE流程的39帧平均耗时142.81→24.77ms/帧、81帧263.80→31.00ms/帧，分别加速5.76倍、8.51倍。两组各30轮AB/BA均更快、P95下降，120帧轨迹逐字节一致，官方整体及逐类指标一致；计时包含CPU/GPU搬运，排除BEVFusion前向。
- **检测模型部署与精度评价：** 完成DEIMv2-S原生及ONNX/TensorRT后端部署，使用官方COCOeval评价完整val2017的5,000图；原生AP由历史YOLOv8s的44.95%提高至50.86%，提升5.91个百分点。完成nuScenes单场景六相机234图预测展示及BEVFusion融合检测部署。
- **跟踪策略与CPU算子优化：** 固定CenterPoint检测完成13组策略对照；在mini单场景39帧中，两阶段贪心关联将连续ID切换46→38，减少17.4%，FP/FN不变。通过复制快路径、批量速度回推、类别广播、二维中心距离矩阵及代价矩阵复用，将同一候选CPU跟踪函数平均耗时2.91→0.77ms/帧，加速3.77倍；30轮配对、完整输出一致。
- **评价与失败案例闭环：** 打通BEVFusion→GRAE的39/81帧官方检测与跟踪评价，构建六相机、点云、BEV及三维轨迹同步复核工作台；将1,910条基线失败记录归并为1,268个事件、84个现象分组，支持距离/置信度筛选、目标定位及片段回放，保存输入、预测和指标绑定证据。

### 项目链接

https://github.com/pyk142857/CARperception

以上是不同任务及独立链路的成果。二维AP来自COCO；身份策略与CPU3.77倍来自CenterPoint输入的mini39帧；GPU5.76/8.51倍来自固定BEVFusion输入的GRAE39/81帧。核心模型使用开源预训练权重，nuScenes成绩限定于mini诊断范围。

## 精简版项目业绩

- 基于nuScenes mini部署多模态检测、三维跟踪、点云分割及地图预测，搭建官方评价与六相机/BEV/3D同步排障闭环；DEIMv2在COCO5,000图的AP44.95%→50.86%。
- 使用TileLang开发FP64批量旋转框3D IoU与GPU NMS，固定BEVFusion→GRAE完整跟踪流程在39/81帧分别加速5.76/8.51倍，含搬运、排除检测前向；各30轮配对，120帧轨迹及官方指标一致。
- 固定CenterPoint输入，两阶段关联使mini39帧连续ID切换46→38、FP/FN不变；NumPy批量计算与代价矩阵复用使同一CPU跟踪函数2.91→0.77ms/帧、3.77倍。

## TensorRT方向的可选补充表述

投递部署/推理优化岗位时可补充或替换模型部署条目：

> 完成DEIMv2-S的ONNX FP32与TensorRT10.6 FP16导出、构建及COCO5,000图评价；在RTX4090、batch1、三张暖缓存图的30轮配对中，TensorRT端到端16.66→7.53ms、约2.21倍，COCO AP下降0.0541个百分点。分析并保留FP16置信度跨阈值变化，作为可选后端，默认保持原生。

计时包含JPEG读取/解码、预处理、H2D、模型、后处理和D2H。严格候选一致性尚未通过；三图性能范围与5,000图精度范围分别报告。ONNX FP32作为正确性/互操作后端，实测更慢。

## 项目设计与各条链路

| 功能 | 实测链路 | 设计与完成范围 |
|---|---|---|
| 数据与几何 | 六相机、LiDAR、标定、位姿和时间戳→帧数据包 | 坐标、框中心、w/l/h、四元数、速度和投影核对；场景边界重置 |
| 二维检测 | YOLOv8s历史对照→DEIMv2-S native FP32 | 同一COCO5,000图官方AP评价；mini234图作展示 |
| 融合检测与学习型跟踪 | 六相机+LiDAR→BEVFusion→冻结检测→GRAE | 39/81帧官方DetectionEval/TrackingEval；GRAE的GPU几何入口已验收 |
| 关联策略与CPU实现 | LiDAR→CenterPoint→两阶段贪心→FastCandidateTracker | 独立mini39帧链路，支撑17.4%及3.77倍 |
| 点云语义分割 | LiDAR→Cylinder3D | 39帧、1,016,188有效GT点、16个非零并集类别，mIoU71.99% |
| 矢量地图 | 六相机→MapTR tiny R50 | 39帧三类折线预测与展示，未作独立地图精度评价 |
| 评价与排障 | 独立预测/GT→官方评价及阈值诊断→Rerun | 官方指标、教学失败统计与显示阈值分开；39/81帧新回放验收通过 |

数据适配、模型推理、评价和回放按模块实现，保存配置、来源与结果。优化时固定输入，先核对语义和实际执行，再测精度与计时，最后决定保留。CenterPoint→FastCandidateTracker与BEVFusion→GRAE分别运行；CPU和GPU加速倍数来自不同对象，不能相乘。历史入口及回放保留，已验证候选通过独立入口复现。

## 成果依据

| 简历表述 | 原始数值与协议 | 直接证据 |
|---|---|---|
| GPU完整GRAE加速5.76/8.51倍 | 39帧142.810612→24.774232ms/帧；81帧263.798978→31.000568ms/帧；每范围30轮AB/BA，均30/30更快，P95下降 | [完整报告](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_gpu_20261013/report.md)、[配对原始记录](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_gpu_20261013/paired_benchmark.json) |
| GPU输出与官方指标一致 | 120帧NMS/网络输入一致，两份轨迹JSON字节一致，整体及逐类指标一致 | [算子校验](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_gpu_20261013/kernel_correctness.json)、[轨迹一致性](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_gpu_20261013/grae_gpu/equivalence.json)、[官方指标一致性](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_gpu_20261013/grae_gpu/official_equivalence.json) |
| 二维AP提高5.91个百分点 | YOLO0.4495358206→DEIM0.5086100690；同一COCO val2017 5,000图、官方COCOeval；不同模型与预训练权重对比 | [YOLO指标](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/results/metrics.csv)、[DEIM指标](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/perception_2025_20261013/deimv2/official_eval/metrics.json) |
| 连续ID切换减少17.4% | 固定CenterPoint，mini39帧，score≥0.25、同类XY<2m；46→38；FP579/FN39；属于教学诊断 | [策略报告](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_optimization/report.md)、[完整对照](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_optimization/metrics.csv) |
| CPU跟踪函数3.77倍 | i9-14900KF，2.910221986→0.772018150ms/帧；30轮交替顺序，双方各1,170帧样本；排除模型/坐标转换/IO/渲染 | [配对报告](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_operator_optimization/report.md)、[摘要](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/tracking_operator_optimization/summary.json) |
| 失败事件工作台 | 1,910条失败→1,268事件→84现象分组；属于组织与复核功能 | [事件报告](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/failure_events/report.md) |
| 可选TensorRT2.21倍 | 三图、batch1暖缓存、30轮；16.664721→7.525682ms；COCO5,000图AP0.5086100690→0.5080690341 | [配对记录](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/perception_2025_20261013/deimv2/tensorrt/paired_benchmark.json)、[精度与限制](https://github.com/pyk142857/CARperception/blob/73efc89338062ed24ff8cb3211e143c4d4354560/perception_lab/reports/perception_2025_20261013/deimv2/tensorrt/report.md) |

## 面试可展开的实现细节

- **GPU热点定位：** 原GRAE的120帧剖析记录452,829次IoU调用，主要耗时来自CPU多边形构造、求交和角点计算。用TileLang0.1.15生成SM89 FP64几何及贪心抑制kernel，通过ctypes在原GRAE进程加载，省去逐帧IPC。CPU仍执行原排序与粗筛，保持先NMS后score过滤及原高度公式。
- **GPU正确性与计时：** 保留FP64/int32/int64；整数索引、顺序和类型精确比较，浮点预算atol1e-8/rtol1e-7。KernelBench使用本项目typed任务扩展；驻留GPU的kernel时间与包含打包/搬运的预处理、完整GRAE时间分别记录。
- **CPU关联算子：** 批量速度回推、类别门控、二维中心距离矩阵与两阶段贪心匹配；减少递归复制和临时数组，复用代价矩阵，保持float32转换位置、相等代价选择顺序及输入/输出隔离。72项测试属于这一阶段。
- **推理部署与排障：** 严格加载完整权重，核对来源、配置和输入哈希，通过模块Hook确认实际执行；评价使用原始预测，展示阈值独立。六相机投影读取各自时刻的ego pose，轨迹跨场景重置。

GPU算子当前capacity256、实测最大199框，默认阈值0.1/1.0/0.3。极小正尺寸可能使上游NMS不终止，已隔离记录reference_failure；候选显式拒绝，不将这些输入宣称为等价通过。

## 已完成实验与保留决定

| 候选 | 完成情况 | 当前决定 |
|---|---|---|
| OptiPMB | 同一冻结BEVFusion39/81帧检测、真实LiDAR点数、原生在线推理及官方评价 | 81帧AMOTA82.30%→79.75%、FN400→456，保留GRAE；作者CenterPoint配置和额外点数信息使该比较属于原生系统比较 |
| BEV pooling索引预计算 | 局部相同输入逐位一致、约5%–10%收益；39/81帧官方检测评价和270组整网配对已执行 | 9帧均仅14–18/30轮更快、未达稳定提速门槛；整网重复波动及重建几何诊断保留，保持上游 |
| MambaFusion→GRAE | 39/81帧真实检测与跟踪评价 | mini_val检测mAP58.90%、AMOTA84.97%，但IDS13→19、FP299→405且重复一致性未通过；保留研究对照 |
| HARP-NeXt与GPU预处理 | 404帧原始点预测和评价 | 同39帧mIoU64.42%低于Cylinder3D71.99%；GPU预处理候选未采用 |
| BEVFusion FP16、GRAE批量D2H | 导出/运行及独立对照已保存 | 前者存在几何/张量诊断问题，后者存在AB/BA顺序依赖，未列为成功优化 |
| Sparse4D、GaussianWorld | 相机时序检测、占用推理扩展已运行 | 模态差异及占用GT缺失限制比较，不列为已证明的优化收益 |

各项nuScenes指标限定于mini诊断，可能与预训练数据重叠。完整val、模型训练/微调、车端实测、规划控制闭环及统一实时吞吐尚未完成。Rerun播放FPS不代表模型吞吐，阶段测试数不合并为总项目测试量。

[项目设计与保留方案](../RETAINED_ALGORITHMS.md) · [GPU完整实验](tracking_gpu_20261013/report.md) · [BEV pooling整网实验](tracking_gpu_20261013/bev_pooling/report.md) · [2025模型总报告](perception_2025_20261013/report.md)

