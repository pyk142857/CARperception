# 现有数据可评估的替代优化方案

日期：2026-10-12  
项目：pyk142857/CARperception  
状态：planned，指令已发布；新模型尚未在本轮下载、编译、推理或评估。  
代码基线：1b702f69601e3c2c46c3a94e52d073e826c6a09b。  
执行指令：[20261012_02_data_matched_optimization.md](../../instruction/20261012_02_data_matched_optimization.md)。

## 结论

把优化主线换为 **D-FINE-S、IS-Fusion、SimpleTrack、SphereFormer**，分别覆盖 2D 检测、相机–LiDAR 3D 检测、3D 跟踪和点级语义分割。它们共用现有数据根目录及可核验真值，不需要先训练或新增占用标注。

旧模型保留作对照。全流程优化优先处理数据准备、CPU 坐标转换和传输，再针对已经通过原生验证的图像子图做后端加速。新模型是否提升、4090 是否能稳定运行，均需本机验收。

## 数据与任务匹配

| 现有数据/产物 | 可做什么 | 本轮算法 | 评估 |
|---|---|---|---|
| COCO val2017 5,000 图与官方 2D 框，配置路径已有记录 | 2D 检测；nuScenes RGB 上辅助展示 | D-FINE-S | COCOeval AP/AP50/小中大目标 AP；先确认本机文件仍在 |
| nuScenes mini 六相机、LiDAR、历史 sweeps、标定与位姿、3D 框 | 相机–LiDAR 融合检测 | IS-Fusion | 官方 DetectionEval mAP/NDS、各类 AP 和 TP errors |
| 同步检测 JSON、关键帧点云、时间/位姿、GT instance_token | 3D 跟踪 | SimpleTrack 2 Hz | 官方 TrackingEval AMOTA/AMOTP/IDS，另列教学 IDSW/FP/FN |
| lidarseg-mini 点级标签：历史核验 404 帧、14,026,208 个标签 | 点云语义分割及远距误差分析 | SphereFormer | 同一原始点顺序下各类 IoU/mIoU/点准确率 |
| 没有确认匹配的密集占用真值 | 占用推理展示 | 保留 GaussianWorld 已有产物 | 继续 blocked_data，不能拿 lidarseg 标签冒充 occupancy GT |

nuScenes 共同比较范围固定为 scene-0061 的 39 帧，以及 scene-0103/scene-0916 的 mini_val 81 帧。分割再补 404 帧全集作为辅助分析，单列范围和预训练重叠局限。相机/雷达/点云的公共数据集介绍不等于本机输入已核验；本轮确定的传感器输入是相机与 LiDAR。

## 算法为何值得换

| 模块 | 论文/实现思路 | 与现有数据的关系 | 首轮限制 |
|---|---|---|---|
| D-FINE-S，ICLR 2025 | Fine-grained Distribution Refinement 逐层细化边框分布；GO-LSD 为训练时定位蒸馏 | 官方 COCO-only 权重直接消费现有 2D 图像；不用重训练 | 固定 S 模型与原生 640×640 评价；公开 T4 速度和论文 AP 都不是本机成绩 |
| IS-Fusion，CVPR 2024 | 从 point-to-grid、grid-to-region 的分层场景融合，到实例上下文引导场景表达 | 直接利用六相机与 LiDAR，已有 3D 标注可以评估 | 官方原生图像 384×1056、10 sweeps 和 0.075 m XY voxel；与旧 BEVFusion 属于整条路线比较 |
| SimpleTrack | 把关联、运动、置信度、轨迹管理作为完整系统分析；本轮使用作者 GIoU+KF+生命周期配置 | 复用封存检测，无模型训练；GT 实例 ID 只进入评估 | 2 Hz 关键帧，固定官方 giou.yaml；不是把原失败的 Hungarian/KF 简单改名 |
| SphereFormer，CVPR 2023 | 球面/径向窗口注意力连接射线附近的稀疏点，增强远距上下文 | 点级分割输出与当前 lidarseg 真值直接对应 | 旧 CUDA/SpTr 项目先过 SM89 编译及严格加载；保留原始点完整覆盖 |

这里按数据、完整权重和可测目标选择算法，年份只是背景。CMT 完整融合权重入口本轮未确认；Pointcept 当前 PTv3 文档提示已发布权重与调整后的结构存在兼容问题；3DMOTFormer 未确认可直接使用的完整预训练权重。它们先不进入默认路线，避免额外训练成为执行前置条件。

## 旧优化为何不再作为主线

- Sparse4D 是纯相机路线，本轮数据中 LiDAR 的几何信息没有进入它的检测网络；它的较低分数不能直接推出“时序模型无效”。改为融合候选后仍按相同 GT 验证，不能提前宣布提升。
- BEV pooling pack4 真实输入正确，但三组 kernel 配对均慢于 upstream，全流程也未稳定提速。保留 upstream，将优化预算转到有明显成本证据的流程。
- 原 TensorRT FP16 的 mini_val mAP/NDS 下降满足预算，部分几何/中间诊断未通过。本轮先保留新模型原生参考，逐子图验证，不复用“总分通过即全部通过”的验收方式。
- GaussianWorld 完成占用推理，却没有匹配占用真值。SphereFormer 解决的是当前可评估的点语义任务，不能据此宣布它的占用性能超过 GaussianWorld。
- 已尝试的简单 Hungarian/KF 跟踪变体出现更多 ID 切换。SimpleTrack 要跑完整作者实现，并固定检测输入，才能区分算法与适配问题。

## 执行和采用顺序

1. 本地预检数据、官方 checkpoint 下载/hash、严格网络匹配、隔离环境、CUDA 扩展和单帧显存。
2. 封存旧基线；补齐 CenterPoint 与 Cylinder3D 的 mini_val 同口径结果。
3. SimpleTrack 先消费已有 BEVFusion 检测，给出成本最低的跟踪对照；其余模型分别单卡运行。
4. 每个模型单帧 → 首/中/末 → 39 帧 → 81 帧。D-FINE 用 COCO 5,000 图评价；两分割模型具备资源后补 404 帧。
5. 原生正确后，每次只改变一项预取/缓存/坐标转换/传输/图像后端因素，30 轮配对测延迟，并重做精度及几何检查。
6. 复用同步回放，输出改善和退化案例、正式 report.md/status.json/metrics.csv，最后提交、推送和核验。

模型比较使用相同数据/GT/评估器，公开各自输入成本；后端优化使用完全相同权重、输入语义和测量边界。工程精度预算为绝对下降不超过 0.005，同时必须通过几何、token、point/index 完整性检查；没有稳定收益就保留参考实现。

本计划不包含新增全量训练。小型数据对照和预训练重叠不能证明车端泛化或量产实时性能。

## 官方来源与资源入口

- [D-FINE 作者代码、S/COCO 配置及权重表](https://github.com/Peterande/D-FINE/tree/956d1709314c2c6a4df6f34de232054578a7449f)，[ICLR 2025 论文](https://arxiv.org/abs/2410.13842)。
- [IS-Fusion 作者代码与完整 checkpoint 链接](https://github.com/yinjunbo/IS-Fusion/tree/86c882a9476fbe046c490349e3d1cd07d23c3b19)，[论文](https://arxiv.org/abs/2403.15241)。
- [SimpleTrack 作者 2 Hz 配置和 nuScenes 执行说明](https://github.com/tusen-ai/SimpleTrack/tree/05c96bb7ed98fc179856f327544612a66c839b5e)，[论文](https://arxiv.org/abs/2111.09621)。
- [SphereFormer 作者 nuScenes 配置及 checkpoint](https://github.com/JIA-Lab-research/SphereFormer/tree/0ff1e0ae84a4280fbecc9cfc3ac3aec41d24eeba)，[CVPR 2023 论文](https://arxiv.org/abs/2303.12766)。
- 项目证据：[TASK_STATE](../../TASK_STATE.md)、[原部署报告](../deployment_4090_20261008/report.md)、[FP16 补查报告](../deployment_4090_20261012_followup/report.md)、[COCO 历史配置](../../configs/coco_val_only.yaml)。

官方入口存在不等于本轮完成下载；实际文件/hash、checkpoint strict load 和硬件兼容以执行产物为准。
