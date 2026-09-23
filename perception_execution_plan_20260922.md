# 自动驾驶感知全流程：部署、调试、评估与可视化执行计划

版本：2026-09-22。用途：交给 Codex，在用户实际工作站上实施。本文是执行设计，不包含已经测得的模型指标。

## 1. 完成目标与范围

最终得到一个可复现的离线感知实验工程：读取真实传感器数据，运行真实预训练模型，完成任务评估和逐场景可视化，再进行推理部署、性能对比，以及一次有对照组的训练改进实验。

默认以单张 NVIDIA 24 GB 显卡作为资源规划起点；执行时先探测实际 GPU、驱动、内存、磁盘和已有数据，再形成资源预算。不能把这个默认值当作已确认的设备配置，也不能预先保证每个模型都能在该显存内运行。

本方案覆盖岗位中的 2D 检测、语义分割、深度、LiDAR 3D 检测、视觉 BEV、相机与 LiDAR 融合、相机与毫米波雷达融合、道路结构、占用预测和多目标跟踪。这里的“完整流程”指这些模块的离线部署与实验闭环；不包含车辆控制、道路实车验收、功能安全认证或端到端规划训练。NeRF、3DGS、Diffusion、World Model 放在后续研究，不占用本轮必需流程。

### 三个可分别验收的里程碑

| 里程碑 | 必须实际完成 | 可以如何描述成果 |
| --- | --- | --- |
| A：完整回放 | M00–M11 对同一批连续场景完成真实推理、输出校验和联合可视化 | 多传感器感知全任务离线回放已跑通 |
| B：部署与评估 | A 加各任务规定的验证集评估、M12 原生推理测速和指定 TensorRT 部署验证 | 各任务精度、时延和部署结果已获得 |
| C：实验闭环 | B 加 M13 对照实验、M14 最终汇总与可复现入口 | 本执行计划全部完成 |

任一必需模块缺失、只显示真值、只安装成功、只下载权重，均不构成 A/B/C 完成。单个模块受阻时继续独立模块，最终如实列出未通过项。

## 2. 数据如何流动

nuScenes 的同一场景提供相机图像、LiDAR、毫米波雷达、标定、时间戳和车辆位姿。它们进入多个并行任务分支；跟踪消费检测结果；可视化程序按 sample token 和时间戳汇总。

| 输入 | 处理分支 | 输出 | 下游 |
| --- | --- | --- | --- |
| 单路相机图像，扩展到六路逐相机运行 | M01 检测、M02 分割、M03 深度 | 2D 框、语义图、米制深度图 | 可视化与各任务评估 |
| LiDAR 当前帧及官方配置要求的历史帧 | M04 PointPillars；M05 CenterPoint | 带类别、置信度、速度的 3D 框 | 对比评估；M05 接 M11 |
| 六路相机及标定、必要的历史状态 | M06 BEVFormer | 3D 框 | 视觉 BEV 对比与回放 |
| 六路相机加 LiDAR | M07 BEVFusion | 3D 框 | 融合评估；接 M11 |
| 相机加毫米波雷达 | M08 CenterFusion | 3D 框与运动相关预测 | 雷达融合评估与回放 |
| 六路相机及模型所需位姿 | M09 MapTR | BEV 中的道路矢量结构 | 道路结构评估与回放 |
| 六路相机 | M10 SurroundOcc | 带语义的 3D 占用网格 | 占用评估与回放 |
| M05 或 M07 的真实检测序列 | M11 CenterPoint 跟踪器 | 跨帧 ID、轨迹、速度 | 跟踪评估与回放 |

不要把 YOLO → 分割 → 深度 → BEVFormer → BEVFusion 全部首尾串联。它们是独立模型或替代方案；深度可视化分支也不是 BEVFusion 必需的外接输入。本轮不擅自把独立分支融合成一个新算法。

## 3. 数据、格式与评估边界

### 3.1 公共回放载体

主数据集采用 nuScenes，先用 mini 做读数和连通性检查，再用官方 trainval 数据完成正式验证。官方 val 的连续场景用于各模型共同回放；mini 的结果仅作为调试证据，不能声称是独立泛化成绩。

- 图像：JPEG；读取后按各模型要求转换 RGB/BGR、归一化、缩放与裁剪，保存预处理参数。
- LiDAR：原始 .pcd.bin 数据由官方工具读取。原始点字段和模型输入字段必须分别声明，不能把 ring 当成历史时间差，也不能套用 KITTI 的四字段读取方式。
- 毫米波雷达：PCD 由 nuScenes devkit 读取，保存点位置、RCS、速度及有效性字段。这里不是激光雷达，也不是所有“4D 成像雷达”都共有的原始数据格式。
- 标定和元数据：官方 JSON 表、camera intrinsic、sensor-to-ego、ego-to-global、各传感器 timestamp。
- 标注：官方 3D 框、类别、实例 ID；MapTR 还需要其官方要求的地图扩展；SurroundOcc 需要与其模型匹配的占用标签。
- 模型专用 infos/pickle：分别由对应版本的仓库生成。新旧 MMDetection3D、MIT BEVFusion 等代码存在数据和坐标约定差异，不共享未经核实的缓存文件。

文件扩展名不是完整的数据契约，坐标、单位、类别、时间戳和预处理也必须记录。此工程的数据格式是公开数据集的格式，不能据此声称某车企内部采用相同存储方式。

### 3.2 正式评估的任务分工

| 模块 | 数值评估数据 | 主要指标 | nuScenes 联合回放用途 |
| --- | --- | --- | --- |
| M01 YOLOv8s | COCO 2017 val，按所选权重支持的类别 | COCO AP、AP50、AP75、分类别结果 | 六路相机检测展示；不把未提供的 2D 标注当作已有 |
| M02 SegFormer B0 | Cityscapes 官方 val 与类别映射 | mIoU、每类 IoU、混淆矩阵 | 展示迁移后的分割结果；没有像素真值就不报该场景 mIoU |
| M03 Metric Depth Anything V2 Small | nuScenes val 的单帧 LiDAR 投影，固定有效掩码 | 稀疏 AbsRel、RMSE、δ1、有效像素数量 | 米制深度与稀疏误差图；明确不是 KITTI 标准成绩 |
| M04–M08 3D 检测 | nuScenes 官方 val，固定 devkit 与配置 | mAP、NDS、各类 AP、误差项 | 同一场景的不同感知方案并排比较 |
| M09 MapTR | 官方协议下的 nuScenes 地图标签 | 官方矢量地图 AP 及分类别 AP | 道路结构叠加；不等价于全部车道拓扑任务 |
| M10 SurroundOcc | SurroundOcc 配套标签及其官方验证划分 | 官方语义 mIoU、几何 IoU 等 | 占用网格、类别和有效区域可视化 |
| M11 跟踪 | nuScenes 官方 tracking val | AMOTA、AMOTP、ID switches 等 | 连续目标 ID、轨迹和速度 |

3D 检测 mAP 的 nuScenes 匹配协议与 COCO AP 不同，不能直接比较。SurroundOcc 标签不能直接替换成 Occ3D 标签；类别、体素范围、可见性和空/未知定义均需要匹配。模型采用各自原生预处理做精度验证，不能为了“公平”强行改成同一分辨率或体素尺寸。

正式评估默认完整官方验证集。下载未完成时可以先做固定 pilot 子集，但结果必须带样本数、子集清单和 pilot 标记，B/C 不得因此判为完成。需要登录、接受许可或手工下载的资产，统一输出缺失清单，再继续可执行的分支。

## 4. 模型清单与逐模块交付

选择公开、具有真实权重和代码的代表模型，目的是覆盖完整任务与工程过程，不把这些模型称为所有车企当前量产方案。Tiny/Small/B0 是预先选择的官方模型变体，不能在运行失败后偷偷替换为自写简化网络。

| 编号 | 默认模型/组件 | 首次实现重点 | 必须产生的结果 |
| --- | --- | --- | --- |
| M00 | nuScenes devkit、数据适配和几何校验 | sensor→ego→global→目标 sensor；多相机时刻对齐 | 数据清单、投影图、坐标校验、共同场景清单 |
| M01 | YOLOv8s 官方预训练权重 | letterbox 还原、类别映射、阈值/NMS | COCO 指标、六视角检测图、预测文件 |
| M02 | SegFormer MIT-B0 / Cityscapes | resize、滑窗、ignore label、trainId | mIoU、分割图、混淆矩阵 |
| M03 | Depth Anything V2 Metric VKITTI Small | 正确的 metric 权重、米制输出、相机 Z 深度 | 稀疏深度指标、带米制色条的深度图和误差图 |
| M04 | PointPillars / nuScenes / SECFPN / 2x | pillarization、点字段、框格式 | 3D 指标、BEV 图、原始与标准化预测 |
| M05 | CenterPoint / nuScenes / voxel 0.1 / SECFPN / circle NMS | 体素化、稀疏卷积、速度、历史点云 | 3D 指标、漏检分析、跟踪输入 |
| M06 | BEVFormer-tiny / R50 官方变体 | 多相机、时序 BEV 状态、场景切换 | 3D 指标、相机 BEV 图、状态重置证据 |
| M07 | MIT BEVFusion / camera+LiDAR / Swin-T | BEV pooling、不同模态对齐、旧版本依赖 | 3D 指标、与 M05/M06 对比、跟踪输入 |
| M08 | CenterFusion / DLA / centerfusion_e60 | radar velocity、投影、DCNv2 | 3D 指标、雷达位置与速度图 |
| M09 | MapTR-tiny / R50 / GKT / 24ep | 道路矢量解码、类别、距离匹配 | 官方 AP、BEV 折线图 |
| M10 | SurroundOcc 官方语义占用模型 | 多视角特征、3D 网格、有效/未知掩码 | 占用指标、3D PLY、BEV 与高度切片 |
| M11 | CenterPoint 官方跟踪器 | 实测检测输入、时间差、状态与类别过滤 | 两组跟踪指标、ID 轨迹视频、ID 错误案例 |
| M12 | 原生推理测速；指定 2D 模型 ONNX/TensorRT | 相同输入/权重的数值与精度回归 | 时延分布、显存、导出日志、部署精度对比 |
| M13 | CenterPoint 两组等预算微调 | 一个有依据的 bad case 假设、固定对照 | 训练曲线、检查点、前后对比、无提升也记录 |
| M14 | 统一回放与实验报告 | 结果关联、覆盖率校验、可重跑入口 | HTML、MP4、CSV/JSON、报告、复现说明 |

模型和权重的 commit、确切文件名、来源 URL、SHA256，在执行时从官方仓库解析并写入锁定清单；本计划中的模型名不代替已经下载验证的资产。MIT BEVFusion 本轮选用检测 checkpoint，不能据此声称它同时输出已训练的 BEV 分割结果。

## 5. 按依赖推进的执行顺序

1. **准备工程与资源**：读现有 AGENTS.md 和 instruction/；自动探测硬件；建立独立工程目录、数据只读引用、模型资产清单和环境方案。
2. **先过 M00**：检查原始数据，保存六相机投影图；在真实样本上核对坐标、尺寸、时间差和 frame token。
3. **M01–M03**：每个模块先单样本，再连续场景，再本任务评估。建立图像分支的输入输出与可视化规范。
4. **M04–M05**：先 PointPillars 定位点云链路问题，再 CenterPoint；所有预测同时保留官方格式和统一回放格式。
5. **M06–M08**：视觉 BEV、LiDAR 融合、毫米波雷达融合。模型环境可以独立，GPU 重任务串行执行。
6. **M09–M10**：道路结构与占用；先确认真值资产，逐步完成推理、标签对齐和正式指标。
7. **M11**：同一跟踪器分别连接 M05 与 M07 的实际预测；固定跟踪参数，比较检测方案对轨迹质量的影响。
8. **先生成里程碑 A 回放**：固定的两个完整 val 场景，展示全部必需分支；分支间用文件关联，不要求同时常驻 GPU。
9. **完成正式评估与 M12**：运行各任务规定的数据范围；部署并比较指定 TensorRT 模型；获得里程碑 B。
10. **M13、M14**：完成一次真实训练闭环，汇总证据和所有状态，生成最终里程碑 C 报告。

每个模块重复“官方入口单样本 → 适配器一致性 → 连续场景 → 评估 → 可视化 → 性能记录”。环境能够导入仅表示安装阶段完成。

不预设“几小时全部跑完”。预检后记录每模块单帧耗时、数据量、显存和空间，再估算总时长。首次编译与下载、全量验证、训练、导出引擎分别列预算，不能与模型纯推理耗时混为一谈。

## 6. 三组必须开展的实验

### 实验 A：输入模态和模型方案对比

在同一 nuScenes val 协议下，比较 PointPillars、CenterPoint、BEVFormer、BEVFusion、CenterFusion 的精度与实测性能；记录每个模型不同的 backbone、输入范围、历史帧数和训练来源。这是不同系统配置的横向对比，不是严格控制网络容量后的传感器消融，也不能把差值全部归因为融合模态。

由 CenterPoint 和 BEVFusion 预测分别驱动同一跟踪器；输出两个独立 tracking run，防止后一个覆盖前一个。

### 实验 B：部署收益

YOLOv8s、SegFormer B0、Metric Depth Anything V2 Small 要完成原生 GPU → ONNX Runtime GPU → TensorRT FP16 的实际运行、数值检查、同协议指标复测和测速。导出不通时调查算子和预后处理，保留失败证据；该项不能改为通过。

对 3D、时序、占用和地图模型，本轮的必需部署等级是原生 GPU 可运行、可复现、已测速。另记录每个模型 TensorRT 的可行性和阻碍；可行性调查不算已经导出成功。含自定义稀疏卷积、BEV pooling 或 deformable attention 的模型需要匹配插件和原始实现。若进一步开展 BEVFusion TensorRT，作为独立扩展记录，不能把部分编码器导出算成完整模型部署。

固定 batch、输入、样本清单、预后处理和权重，报告 p50/p95、峰值显存以及精度差异。计时规则见执行指令。这里不承诺 INT8；若扩展 INT8，校准集只能来自训练划分，且单独评估精度损失。

### 实验 C：一次 bad case → 假设 → 微调 → 复评

默认选 CenterPoint：

- 先运行真实预训练基线并保存验证集预测，分类统计远距离、少点目标、类别等失败。
- 从训练划分选择固定 80 个完整场景作为小规模微调集，不从 val 取训练样本。选取规则、scene token 和种子入档。
- 建立两组同预算实验：A 为常规微调，B 仅改变一个由训练数据分析支持的因素，例如一项类别采样策略。具体因素与假设先写入 experiment_card。
- 默认每组 1,000 次 optimizer update，微批量 1、梯度累积 4；如原生实现不支持，先实现并检查优化器步数，不能把 micro step 冒充 optimizer update。优化器种类遵循原配置，初始学习率以原配置的 0.1 倍作为起点，并在两组训练前固定。按实际显存决定是否启用经过验证的 AMP。
- 两组使用同一个初始 checkpoint、相同数据、种子和训练预算，保存参数确实更新、loss、学习率、检查点和恢复训练证据。
- 最终在同一个官方 val 上比较初始基线、A、B。val 用于最终比较，不以反复试验挑最优结果；训练范围与预算必须公开。

这是有限预算的真实微调实验，不是从零复现论文。指标没有提升也可以构成完成的实验；必须明确“未观察到提升”，不能虚构收益。

## 7. 验收证据

每个模型的目录至少包含：环境/版本清单、模型/配置哈希、输入场景清单、实际命令与日志、原始预测、标准化预测、指标 JSON、运行统计、可视化样例、状态文件。

最终交付：

- results/metrics.csv：实测结果与适用的数据协议、样本数、单位、方向。
- results/performance.csv：模型级和整套串行回放的独立性能记录。
- results/status.json：每模块 native、evaluation、visualization、export 等字段的完成状态和证据路径。
- reports/final_report.md：结论、失败案例、训练对照、部署收益与局限。
- reports/index.html：本地可打开的总览，所有指标与图片指向真实文件。
- outputs/replay/：两个完整场景的 MP4、逐帧索引和实际传感器时间信息。
- outputs/geometry/、outputs/occupancy/、outputs/tracking/：投影图、PLY/切片、轨迹与错误案例。
- configs/locked/、envs/、logs/、checkpoints/manifest.json：足以恢复运行的配置、版本与资产证据。
- README.md：从已准备数据到重跑实验的真实已验证命令。

论文参考值可以单列，但必须标明 source=paper_reference。任何主结果都不能用论文数字、零值、随机输出或手写示意图填充。

## 8. 官方资源入口

以下是模型和协议选择依据；确切依赖与可下载资产由执行时再次核验并锁定。

1. nuScenes 数据与 SDK：https://www.nuscenes.org/nuscenes ，https://github.com/nutonomy/nuscenes-devkit
2. nuScenes 检测评估：https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md
3. nuScenes 跟踪评估：https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/tracking/README.md
4. YOLOv8：https://docs.ultralytics.com/models/yolov8/ ，https://github.com/ultralytics/ultralytics
5. SegFormer 实现及配套模型表：https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer
6. Metric Depth Anything V2：https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth ，https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small
7. PointPillars 模型表：https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars
8. CenterPoint 模型表：https://github.com/open-mmlab/mmdetection3d/tree/main/configs/centerpoint
9. BEVFormer：https://github.com/fundamentalvision/BEVFormer
10. MIT BEVFusion：https://github.com/mit-han-lab/bevfusion
11. CenterFusion：https://github.com/mrnabati/CenterFusion
12. MapTR：https://github.com/hustvl/MapTR
13. SurroundOcc：https://github.com/weiyithu/SurroundOcc
14. CenterPoint 跟踪入口：https://github.com/tianweiy/CenterPoint/blob/master/docs/NUSC.md
15. COCO 数据：https://cocodataset.org/#download
16. Cityscapes 数据与评估工具：https://www.cityscapes-dataset.com/ ，https://github.com/mcordts/cityscapesScripts
17. TensorRT 导出与性能：https://docs.ultralytics.com/integrations/tensorrt/ ，https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html

## 9. 如何交给 Codex

将整个执行包放进目标工程；把 instruction/ 内的主指令作为本次任务入口，并要求读取 configs/execution_plan.yaml。

主指令对“做什么、怎样算完成”作完整规定；YAML 是待实现程序读取的执行计划，不是已存在的可运行流水线。本次交付没有宣称已经训练、运行或导出任何模型。
