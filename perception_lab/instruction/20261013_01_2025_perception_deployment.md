# 2026-10-13：仅采用 2025 年方法的感知部署与优化

## 目标与状态

BOSS 最新要求：“找点2025的，旧的不考虑。”
本指令替代 20261012_02_data_matched_optimization.md；停止旧方案候选部署，不运行旧 phase all。
本轮四项候选为 DEIMv2-S、MambaFusion、GRAE-3DMOT、HARP-NeXt。旧模型只保留历史产物供参考，不重新训练或为凑对照重跑。
当前状态 planned：选型、作者源码/配置/权重入口核查已完成；尚未执行本机下载、编译、GPU 推理或指标评价。

项目 pyk142857/CARperception，本地 /home/minglei/Desktop/CAR，单张 RTX 4090 24 GB。
读取 AGENTS.md、README.md、TASK_STATE.md 后，在 codex/20261013_2025_perception 分支推进。保留未提交工作、数据、环境及旧报告。

| 任务 | 方法、公开时间依据 | 复用数据 | 评价 |
|---|---|---|---|
| 2D 检测 | DEIMv2-S，arXiv v1 2025-09-25 | COCO val2017；nuScenes 六相机展示 | COCO AP/AP75/APs，原生与后端对照 |
| 融合 3D 检测 | MambaFusion，arXiv v1 2025-07-06、ICCV 2025 | 六相机、LiDAR、sweeps、标定、位姿、3D 框 | 官方 mAP/NDS、各类误差 |
| 3D 跟踪 | GRAE-3DMOT，CVPR 2025 公开论文 | 已有封存检测、新融合检测、时间/位姿、GT实例ID | 官方 AMOTA/AMOTP/IDS |
| 点云语义分割 | HARP-NeXt，arXiv v1 2025-10-08、IROS 2025 | LiDAR 与 lidarseg-mini 标签 | 原始点级 IoU/mIoU/准确率 |

年份以公开论文为依据；GRAE 未确认单独 arXiv 首发日，不虚构日期。
不纳入“2025会议但2024已首发”的方法。作者实现中的通用组件/预处理依赖不算旧模型候选。
无需重新训练、全量 nuScenes、Cityscapes 或新增占用标注。点语义与 occupancy 分开；既有占用缺GT状态保留。

## S0：固定资源、数据与环境

固定作者源码：

| 方法 | 仓库 commit | 作者配置/入口 |
|---|---|---|
| DEIMv2-S | Intellindust-AI-Lab/DEIMv2@1d2ca42171570c713e78fc6a766ec5104b7f4724 | configs/deimv2/deimv2_dinov3_s_coco.yml；tools/inference/torch_inf.py |
| MambaFusion | AutoLab-SAI-SJTU/MambaFusion@d85c2feeb19207b20e42d50421afb11c095f3efa | tools/cfgs/mambafusion_models/mamba_fusion.yaml |
| GRAE-3DMOT | altkddhfcjs/GRAE-3DMOT@63def8bde5e199a4e77fdf4fab76a4b3511fe132 | config/val.json；validation.py、trainer/predictor.py |
| HARP-NeXt | SamirAbouHaidar/HARP-NeXt@532df2659700ae870c0e0d4137a4d35bcc9903b9 | configs/net/harpnext-nuscenes.yaml；configs/main/main-config.yaml |

官方完整权重入口：
- DEIMv2-S：https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view?usp=sharing
- DEIMv2 作者 Hub：https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO
- MambaFusion：https://drive.google.com/drive/folders/1TqvpIHA7plzoFdnGWvFgVYr45bgz-nQ3?usp=sharing
- GRAE-3DMOT：https://drive.google.com/drive/folders/1V-rSzUsOtOY-hoxQET8r9unDnUudks4I?usp=sharing
- HARP-NeXt：https://cloud.minesparis.psl.eu/index.php/s/eUOcaEXZRLNfV1e

配套资源：
- DEIMv2-S 配置路径 ckpts/vitt_distill.pt 的作者 ViT-Tiny 蒸馏初始化：https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing 。完整 checkpoint 严格覆盖 backbone 时，沿作者推理构造避免不必要重复加载；构造器确实要求文件则下载匹配资源。
- MambaFusion Hilbert 模板：https://drive.google.com/drive/folders/1huzbUotusvePvsR-DyGJc6qfX0Cwubvd?usp=sharing ，或按作者 tools/hilbert_curves/create_hilbert_curve_template.py 生成；保存参数、shape及hash。
- GRAE 作者于 2025-12-22 公布论文 checkpoint 和处理后数据，只取所需 checkpoint/config，不用作者 full-val 产物冒充本项目结果。
- HARP 下载 nuScenes-32×480 checkpoint 目录，核对 ckpt_best.pth/ckpt_last.pth，按作者 checkpoint["net"] 加载。

资源入口存在不等于已经下载。记录实际文件名、字节数、SHA256、源码/配置/依赖版本、严格加载 missing/unexpected keys、LICENSE 路径。原生与 Hub 封装固定选一条主路径，不混用配置或 state_dict。

本地预检：
1. 从已有配置解析根目录，确认 COCO val2017 5,000 图及 instances_val2017.json。历史 configs/coco_val_only.yaml 的 train=val 只供评价，禁止据此训练。
2. 固定 scene-0061 39 帧；mini_val 为 scene-0103/scene-0916 81 帧。核验相机顺序、sample/sample_data token、位姿/标定、时间戳、实际 sweeps 与文件 hash。
3. 重新核验 lidarseg-mini 所用帧的原始点/标签数量；历史 404 帧/14,026,208 标签是库存记录，不是新推理成绩。
4. 当前已确认相机与 LiDAR，不新增未经核验的 radar 输入；缺数据模块标 blocked_data，继续其余模块。
5. 模型依次严格加载→单帧→首/中/末三帧→39/81帧；batch=1，记录GPU/RAM峰值、dtype/TF32、模块调用和非有限值。
6. 隔离环境：DEIMv2 pinned requirements 为 torch2.5.1/torchvision0.20.1；Mamba 作者 torch2.1/cu118；HARP 作者 torch2.5.0/cu118；GRAE 作者 torch1.9/cu111，需验证4090兼容。不得替换现有 TensorRT8.6/SpConv1。
7. CUDA扩展采用支持SM89的工具链，记录 nvcc/runtime/Torch/GCC/SpConv/torch-scatter/Mamba 版本及实际算子 smoke test。
8. HTML假下载、缺完整权重、strict load失败、OOM、关键模块零调用分别记录阻塞，继续其余模块。禁止随机权重、strict=False、静默改宽度/分辨率或把CPU替代写为4090成绩。
9. GPU推理和性能计时串行，不偷偷启动训练或全量下载。构建/API/路径与mini入口可适配，数学或权重布局修改必须独立验证。

产物先写 reports/perception_2025_20261013/preflight.json、resource_manifest.json、environment.md。

## S1：DEIMv2-S，2D 检测与后端优化

- 固定 DINOv3 蒸馏 ViT-Tiny + Spatial Tuning Adapter（STA），embed_dim=192、interaction_indexes=[3,7,11]、四层decoder、640×640、300 queries。S不是HGNetv2-N，也不是直接运行大教师。
- 展开所有include，沿validation Resize/float scale/ImageNet Normalize，关闭训练增强。作者采用 EMA module 或 model 的选择方式，严格加载 train-form 再 deploy()。
- Hook证明 backbone、STA多尺度输出、HybridEncoder、decoder运行，保存形状/次数/权重覆盖。
- COCO原始image_id与sparse category_id、原图xywh逆变换，保存完整预测；可视化阈值不影响评价。
- 作者 faster-coco-eval 结果可另存；共同口径使用 pycocotools COCOeval，对同一JSON保存 AP/AP50/AP75/APs/APm/APl/AR。
- 跑现存5,000图；nuScenes六相机仅展示2D结果，没有匹配2D真值不报该范围AP。
- 原生正确后执行作者 export_onnx.py --check，核查实际输入输出/shape与后处理，再测后端。
- 作者明确 FP16 TensorRT要求 >=10.6；10.4有错误输出的已知问题，现有8.6不能直接当已支持后端。独立环境构建并保存版本/engine/build logs；受阻继续原生/ONNX FP32。
- 同权重后端AP绝对下降预算0.005，并检查框还原、非有限值和中间数值/几何；不能只看最终AP。

## S2：HARP-NeXt，点语义及GPU预处理

- 固定 nuScenes-32×480 模型：XYZ/intensity、range_H=32/range_W=480、FOV up10°/down−30°、16类。Hook证明 Conv-SE-NeXt、range-point融合和逐点head运行。
- 作者 processor 使用 nuscenes_files.npz，并硬编码 train28130/val6019/test6008。生成 mini 同schema路径/token列表与真实数量校验，不能读full-val列表。
- eval模式关闭TTA/训练增强/Instance CutMix。先CPU预处理参考，再 netconfig.preproc.gpu=true 作作者GPU候选；两边batch=1。
- RangeInterpolation可能追加合成点：保存原始点数N、追加点数及映射，网络使用作者所需输入，输出恢复至每个原始点。最终mIoU只计原始有效GT点；插值点不计原始观测/额外GT。
- 作者manager的confusion直接消费处理后的pt_labels，是否包含插值点需实测核验；可保留原生辅助统计，共同结果必须注明 original-points-only。
- 原始32类GT按官方/作者表到0–15，显示统一1–16，noise ignore；逐类名称/映射/点顺序共同核对。
- GT变更不应改变网络特征或预测：原始点固定、标签替换/置空只影响评价，GT投影labels不得进入特征。
- 顺序跑39/81帧，资源完整后补现存404帧全集辅助结果。保存点数、confusion、各类IoU/union、union>0 mIoU、有效点准确率，缺类N/A。
- CPU/GPU A/B检查投影碰撞选择、point index=0、边界、插值顺序、点覆盖、重复稳定性。
- 作者4090“100 FPS、77.1 mIoU”为外部基准，本机重新测含预处理/传输/逐点回传的端到端，可视化另外计时。

## S3：MambaFusion，相机–LiDAR 融合检测

- 保留原生六相机256×704、10 sweeps、XYZ/intensity/timestamp、范围[-54,-54,-5,54,54,3]、作者实际体素参数与200 proposals。
- 使用新模型自己的原始数据pipeline；旧同尺寸张量不可直接复用。timestamp不是ring，沿作者sweep loader，记录实际历史数量和不足历史处理规则。
- OpenPCDet VERSION=v1.0-mini/mini infos_withimg/当前token列表适配。eval不做GT sampling/velocity aug/训练增广，不生成训练GT database或150GB共享缓存。
- 实测 selective_scan、作者 mamba_diffv/Mamba、自定义扩展及必需voxel/IoU组件。不得用attention/MLP代替Mamba。核验Hilbert模板、扫描次序、shape/hash。
- resolved YAML中USE_*开关与实际Hook逐一对应：LiDAR高度编码、相机与点云分支、Hybrid Mamba局部/全局融合、视图变换、head。关键开关开启而调用为零应失败。
- 保存shape/dtype/时间，39/81帧真实预测后官方DetectionEval，置信度排序匹配、同split/GT、10类/class-range/max-box规则；显示阈值不裁评价预测。
- 核对 global/LiDAR、wlh/XYZ尺寸、yaw/四元数、速度旋转、timestamp及test shuffle的重复噪声。annotations只供独立评价。
- 另存原始attributes；与历史参考比较时用相同策略，禁止从GT填属性。
- 首轮不强制全图ONNX/TRT；优先原生热点、稀疏布局与scan优化。AMP在扩展支持、原生正确、几何/精度过关后独立验证。

## S4：GRAE-3DMOT，学习式关联

- 完整作者checkpoint严格匹配 config/val.json 的 GRAE in_channels128/layers3/单卡；下载资源若有不同匹配config，以实际记录为准。
- 运行空间关系编码、时空关系编码和距离感知融合/关联分数，保存权重/Hook。SimpleTrack只作为作者NMS/坐标依赖，不部署旧跟踪器候选。
- 作者 predictor 硬编码 v1.0-trainval/'val'/相对路径，convert_dataset默认train并带GT分支。新增mini/root/split/输入JSON显式参数，只准备推理字段。
- 输入真实封存检测或MambaFusion预测：score/class/size/yaw/rotation/velocity/timestamp/scene/sample，核对作者7类和坐标/单位。
- GT tracking_id、未来框、关联标签不能进入模型/关联/阈值。val reader如要求tracking_id字段，用非GT sentinel并验证未用于预测；ID由tracker生成。
- 固定两套检测输入独立实验：封存参考、新MambaFusion native。保存各自输入hash，不能跨检测变化把全部收益归给tracker。
- 使用真实2Hz关键帧/timestamp差，跨scene reset；覆盖空检测、首末帧和断点恢复。
- 原生入口 validation.py -c config/val.json -r <checkpoint> -d cuda:0 -o <output>，适配后给出实际工作目录与可运行命令。
- 在39帧冻结配置，再跑mini_val81帧，用官方TrackingEval计算AMOTA/AMOTP/IDS/FP/FN。教学score>=0.25、同类中心<2m的IDSW另列。
- 给拥挤近邻、遮挡/短漏检、轨迹中断的改善与退化证据，不用作者processed full-val成绩代替。

## S5：共同优化、回放与交付

原生正确后逐项A/B，按实际绝对毫秒排序：
1. HARP作者CPU/GPU预处理，语义和原始点覆盖一致。
2. DEIMv2 PyTorch→ONNX→独立TensorRT>=10.6，固定权重/640×640/后处理。
3. 常驻模型、有界CPU预取、metadata预解析、批量坐标/速度变换、buffer复用、pinned memory/H2D；位姿按帧绑定，速度不加平移。
4. Mamba scan/serialization/voxel reorder只有profile证明热点才改内存布局/launch/kernel；记录真实输入及等价检查。
5. GRAE批量几何/attention及减少D2H同步，保持输入、状态和关联语义。

性能：单GPU串行、相同缓存/输入/权重、预热后30轮交错配对；P50/P95/均值/波动、绝对毫秒和峰值显存。
分开cold-start、读取/解码/预处理、H2D、GPUforward、decode/D2H、坐标标准化、可视化。GPU用同步正确CUDA event，端到端wall clock从原始数据到可消费预测。
不拿预存特征的候选与含磁盘/预处理的参考直接比。
同模型工程预算：2D AP、3D mAP/NDS、点级mIoU最大绝对下降各0.005，且几何/index/token/point顺序/非有限值检查通过。跟踪等价优化应保持关联/轨迹；行为变化单列并重评估。
没有稳定收益就保留该新模型native并说明，不自动回退旧方法冒充新候选通过。

实现 configs/perception_2025.yaml 与 tools/run_perception_2025.py 或等价入口；支持 preflight/deimv2/harpnext/mambafusion/grae3dmot/optimize/visualize/summarize、单阶段和恢复。
all只含本轮四方法，不仅改模型名称，实际网络/权重/Hook与开关必须对应。上述为待实现接口，README命令须实测可执行。

复用9094/Rerun，同token同步六相机、原始点云、检测框、ID/轨迹、原始点语义和独立GT。核对首/中/末、坐标、点数、ID颜色及旧图层清理；缺帧显示缺失，不用前帧假填。
产物 reports/perception_2025_20261013/：保留plan.md，执行后生成report.md/status.json/metrics.csv/json、resource_manifest.json/environment.md/resolved_configs/official_eval/profile/badcases/visualization。
状态真实区分planned/running/passed/partial/blocked_resource/blocked_data/failed；记录实际frame/image/point数、时间、采用决定和阻塞原因。
指标来源 new_measured/reused_archive/external_paper 分开；外部数字不填本机行，不适用留空。mini/预训练可能重叠、模型原生输入预算和计时边界如实披露。
完成必要的类别/坐标/点映射/reset/权重/模块执行检查及相关测试，实际推理、官方评价与回放验收后更新README/TASK_STATE。
按AGENTS既有授权提交代码/配置与可分享结果、推送并读回核对远端SHA和报告。数据、checkpoint、环境、第三方源码、大缓存留本地，仓库保存来源/hash/生成方式。
最终报告真实提升/退化、四模块状态、失败原因、采用决定与链接；本轮选型/指令发布不能写成已完成部署。
