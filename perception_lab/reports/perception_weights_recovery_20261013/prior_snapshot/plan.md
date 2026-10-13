# 2025 方法限定的感知优化方案

日期：2026-10-13  
状态：planned；完成选型与官方资源核查，未执行本机下载、编译、GPU实验。  
项目基线：main@ae6e4af25ba1b54ffaf9edc9c4856a7d6b466031。  
执行：[20261013_01_2025_perception_deployment.md](../../instruction/20261013_01_2025_perception_deployment.md)。

## 当前候选

BOSS要求“找点2025的，旧的不考虑”。本轮改为 DEIMv2-S、MambaFusion、GRAE-3DMOT、HARP-NeXt。
上一份20261012_02候选方案已被替代，不继续其部署。旧报告与封存预测保留作历史证据；新候选无权重/资源时标具体阻塞，不自动回退旧模型。

| 方法 | 时间依据 | 核心改进 | 现有数据适配 |
|---|---|---|---|
| DEIMv2-S | arXiv v1：2025-09-25 | DINOv3蒸馏ViT-Tiny、STA多尺度适配、改进Dense O2O及轻量decoder | 现存COCO val2017定量评价；nuScenes六相机辅助展示 |
| MambaFusion | arXiv v1：2025-07-06；ICCV2025 | 保留高度的LiDAR编码改善跨模态对齐，Hybrid Mamba进行局部/全局融合 | 六相机+LiDAR+sweeps/标定/位姿，官方3D框评价 |
| GRAE-3DMOT | CVPR2025公开论文 | 空间/时空关系编码、距离感知融合，学习关联分数 | 消费现有检测JSON与时间/位姿，官方实例ID只供评价 |
| HARP-NeXt | arXiv v1：2025-10-08；IROS2025 | 快速预处理、Conv-SE-NeXt、多尺度range-point融合 | LiDAR和现有点语义标签，原始点级评价 |

GRAE未确认单独arXiv首发日；以作者/CVF的2025公开论文为依据，不拿GitHub更新时间充当论文首发。
DEIM原版、D-FINE和MCTrack存在2024公开版本，因此即使正式会议为2025，也不进入本次候选。
所有方法都可复用旧基础组件；它们的年份以新方法公开时间为准。后续维护补丁不改变其首发年份。

## 官方实现与实际部署检查

| 方法 | 配置/权重核查结果 | 本机首轮需解决 |
|---|---|---|
| DEIMv2-S | 作者S/DINOv3配置、原生完整权重、作者Hub入口已公布；S不是HGNetv2-N | 640×640/300queries，strict EMA/model加载与STA执行；FP16要求独立TensorRT>=10.6 |
| MambaFusion | 作者配置、完整checkpoint文件夹、Hilbert模板入口均公布 | Mamba/selective_scan扩展、模板与扫描顺序、mini infos_withimg、真实融合分支/显存 |
| GRAE-3DMOT | 作者2025-12-22公布匹配论文checkpoint；val.json为单卡GRAE128维/3层 | 旧Torch/CUDA兼容；默认full-trainval及GT转换路径须改为mini推理输入 |
| HARP-NeXt | 作者nuScenes-32×480配置和checkpoint入口公布；有4090测试 | mini列表/计数断言适配；原始点与插值点隔离；CPU/GPU预处理等价 |

作者外部基准：DEIMv2-S为50.9 COCO AP/约9.7M参数；MambaFusion为72.7mAP/75.0NDS（nuScenes val）；GRAE为73.7% AMOTA（CenterPoint检测，nuScenes val）；HARP为77.1mIoU/100FPS（作者4090设置）。
这些数字属于 external_paper，不是本项目测量，也不与mini历史数字直接相减。

具体已发现的接口问题：
- DEIMv2官方README明确FP16使用TensorRT>=10.6，10.4可出错，现有项目8.6保持隔离。
- MambaFusion配置默认训练GT sampling、共享缓存路径和full infos；只执行eval，不生成训练GT database。
- GRAE predictor默认v1.0-trainval，convert默认train并读取GT；mini推理仅准备模型必需字段，不能用GT ID/未来框关联。
- HARP processor默认nuscenes_files.npz并断言28130/6019/6008帧；mini必须生成独立列表和真实计数。
- HARP RangeInterpolation追加点，而原生manager直接统计处理后labels；需核验统计范围，共同mIoU严格恢复至原始有效GT点。GT只参与评价，标签替换不得改变模型特征/预测。

## 数据、顺序与验收

固定 scene-0061 39帧，mini_val=scene-0103/scene-0916 81帧。点语义资源齐全后补现存404帧辅助分析。COCO5,000图先确认文件仍在本机。
不新增全量下载或训练，不要求occupancy真值。点语义标签不能替代密集占用GT。

先预检与单帧，再分模块执行：
1. DEIMv2-S原生及COCO评价、HARP原生/作者GPU预处理对照。
2. MambaFusion六相机–LiDAR检测39/81帧，官方DetectionEval。
3. GRAE分别消费封存参考检测和新融合检测，官方TrackingEval；保持检测输入hash和真实2Hz时间。
4. 在每个新模型原生正确后，再做后端/流程优化与同步回放。

资源入口核查不代表已下载或通过4090。每项严格加载、SM89算子smoke test、峰值显存、关键模块Hook先过关；独立阻塞不停止其他可运行模块。

同模型优化采用相同权重/输入/缓存口径，预热后30轮配对，公开P50/P95/绝对毫秒和显存。
2D AP、3D mAP/NDS、原始点mIoU的最大绝对下降各0.005为工程预算，同时几何/index/token/点覆盖/数值检查须过关。
有稳定收益才采用优化候选；否则保留该2025模型的native，并记录partial/失败原因。
执行后生成report.md、status.json、metrics.csv/json、official_eval、profile、badcases与回放证据。本plan.md始终作为计划保留，不改写为成绩。

## 官方来源

- [DEIMv2论文](https://arxiv.org/abs/2509.20787)，[作者实现](https://github.com/Intellindust-AI-Lab/DEIMv2/tree/1d2ca42171570c713e78fc6a766ec5104b7f4724)，[作者S权重](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO)。
- [MambaFusion论文](https://arxiv.org/abs/2507.04369)，[作者实现](https://github.com/AutoLab-SAI-SJTU/MambaFusion/tree/d85c2feeb19207b20e42d50421afb11c095f3efa)。
- [GRAE-3DMOT论文](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_GRAE-3DMOT_Geometry_Relation-Aware_Encoder_for_Online_3D_Multi-Object_Tracking_CVPR_2025_paper.html)，[作者实现及资源](https://github.com/altkddhfcjs/GRAE-3DMOT/tree/63def8bde5e199a4e77fdf4fab76a4b3511fe132)。
- [HARP-NeXt论文](https://arxiv.org/abs/2510.06876)，[作者实现及资源](https://github.com/SamirAbouHaidar/HARP-NeXt/tree/532df2659700ae870c0e0d4137a4d35bcc9903b9)。
