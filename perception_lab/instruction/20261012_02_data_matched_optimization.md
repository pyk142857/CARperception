# 2026-10-12：用现有数据替换优化主线

## 任务与当前状态

执行对象：pyk142857/CARperception，本地工作目录 /home/minglei/Desktop/CAR，单张 RTX 4090 24 GB。
本指令发布时的代码基线为 main@1b702f69601e3c2c46c3a94e52d073e826c6a09b。
本轮状态为 planned：已完成方案和资源入口核对，尚未执行下载、编译或 GPU 实验。不得把本文件当成实验报告。

BOSS 的要求是“换一套当前数据都能用上的优化算法”。本轮采用：

| 模块 | 新候选 | 可复用输入及真值 | 比较对象 |
|---|---|---|---|
| 2D 检测 | D-FINE-S，COCO-only 权重 | 已有 COCO val2017 图像及 instances_val2017.json；nuScenes 六相机用于展示 | YOLOv8s |
| 相机–LiDAR 3D 检测 | IS-Fusion | 六相机、LiDAR、已有历史 sweeps、标定、位姿、时间戳及 3D 框 | BEVFusion 原生参考、CenterPoint |
| 3D 多目标跟踪 | SimpleTrack 作者 2 Hz GIoU 配置 | 上述检测 JSON、关键帧点云、位姿及 GT instance_token | PubTracker 原始及已实现的 greedy 候选 |
| 点云语义分割 | SphereFormer | 原始关键帧 LiDAR 与现有 lidarseg-mini 点级标签 | Cylinder3D 原版 SpConv 1 |
| 推理流程 | 有界预取、常驻模型、元数据复用、批量坐标转换；条件通过后优化图像后端 | 同一模型、权重及原始输入 | 每个新模型的原生实现 |

这些算法均有官方实现和已公布的预训练权重入口；本轮尚未确认本机下载成功、编译兼容或显存峰值，必须先做 S0。
本轮不以训练、下载 nuScenes 全量 trainval 或补标新任务为前置条件。
保留 GaussianWorld 已有推理与 blocked_data 状态；SphereFormer 输出点级语义，不能称为占用网络或用点标签计算密集 occupancy IoU。
SegFormer、深度、MapTR 既有展示继续保留；没有匹配真值的模块不虚构精度。

## 执行原则

1. 先读取 AGENTS.md、README.md、TASK_STATE.md、原部署报告及补查报告，再实现本轮配置与编排。遵守指令、报告目录约定。
2. 在独立工作分支 codex/20261012_data_matched_optimization 上推进；遇到已有同名分支先检查状态，不覆盖未提交工作。
3. 保存旧输出、权重、虚拟环境与默认推理路径。新候选通过验收才增加可选择的默认配置；未通过仍发布失败证据。
4. 四个模块独立记录 passed / partial / blocked_resource / blocked_data / failed。一个模块受阻继续执行其余可运行模块，不把随机初始化、替代数据或静默跳过写为成功。
5. GPU 推理和性能计时串行执行；禁止同时跑多个模型后声称单模型延迟。允许 CPU 准备、只读检查并行。
6. 不修改旧报告的历史数字。区分本轮重新推理、重评估与复用封存产物。权重、数据、环境、第三方源码及大文件继续留在本地。
7. 不调用旧 --phase all。下文的新编排命令是待实现接口，必须先实现并验证，不能假称当前仓库已有此命令。

## S0：输入、资源、环境预检

输出到 reports/data_matched_optimization_20261012/preflight.json、resource_manifest.json 和 environment.md。

### 本地数据

- 从现有配置及部署 inputs/summary.json 解析实际数据根目录，不硬编码猜测路径。
- nuScenes：核验 v1.0-mini 元数据、六路相机顺序、关键帧 LiDAR、历史 sweep 引用、每帧校准和 ego_pose。
- 固定 scene-0061 共 39 帧；mini_val 为 scene-0103 和 scene-0916，共 81 帧。保存 scene/sample/sample_data token、原始文件 SHA256、时间戳和 split manifest。
- 实际遍历历史 sweep 引用，记录每帧可用数量及作者的短历史处理规则。IS-Fusion 原生配置 sweeps_num=10，旧 BEVFusion 输入策略不同；不得把不存在的历史帧伪造成真实观测，不得强行共用旧预处理张量。
- lidarseg-mini 历史核验为 404 帧 / 14,026,208 个标签。本轮重新核对所用帧的原始点数、标签数、映射和文件可读性；GT 只进入评估与可视化分支。
- COCO：读取 configs/coco_val_only.yaml，确认 val2017.txt 对应 5,000 张现存图像，读取官方 instances_val2017.json 并核对 image_id、尺寸及 category_id。原配置 train 指向 val 只是历史评估配置，本轮不得据此训练。
- 若 COCO 数据已被本机清理，D-FINE 标记 blocked_data，先完成 nuScenes 三模块；记录缺失路径，不能换图后声称 COCO 全量成绩。
- 当前确认的感知输入是相机与 LiDAR；不得把未经本机核验的 radar 文件写成已使用输入。

### 官方资源与固定版本

| 模型 | 作者仓库与固定提交 | 作者配置 | 权重 |
|---|---|---|---|
| D-FINE-S | Peterande/D-FINE@956d1709314c2c6a4df6f34de232054578a7449f | configs/dfine/dfine_hgnetv2_s_coco.yml，递归展开 include | https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth |
| IS-Fusion | yinjunbo/IS-Fusion@86c882a9476fbe046c490349e3d1cd07d23c3b19 | configs/isfusion/isfusion_0075voxel.py | https://drive.google.com/file/d/1mY2juJ2n0Dw5NWDSraZXrdU1RwkE-40h/view?usp=sharing |
| SimpleTrack | tusen-ai/SimpleTrack@05c96bb7ed98fc179856f327544612a66c839b5e | configs/nu_configs/giou.yaml | 无学习权重 |
| SphereFormer | JIA-Lab-research/SphereFormer@0ff1e0ae84a4280fbecc9cfc3ac3aec41d24eeba | config/nuscenes/nuscenes_unet32_spherical_transformer.yaml | https://mycuhk-my.sharepoint.com/:u:/g/personal/1155154502_link_cuhk_edu_hk/Ebj08nZvE5lPpRn1ALgkcKwBjEQ5lrQFhx-yR2cbi9Cy-A?e=D3N3ge |

- 按作者链接获取到本地，验证下载内容不是 HTML 登录页，记录重定向后的来源、文件字节数和 SHA256。
- 保存所用完整配置、递归 include、源码 commit、LICENSE 路径以及环境依赖。作者论文数字只作外部背景，不能填入本轮指标表。
- D-FINE 使用 COCO-only S checkpoint；不要无记录改为 Objects365 预训练版本、N/M/L 型号或社区转换权重。
- IS-Fusion 必须是完整融合 checkpoint；仅 image pretrain 不算融合模型。依据作者推理方式选择 model/EMA 状态，记录 missing_keys/unexpected_keys，严格检查网络参数及持久 buffer。训练时的预训练加载入口不得在完整 checkpoint 推理时覆盖其权重。
- SphereFormer 必须匹配 nuScenes 16 类结构及官方 checkpoint，禁止换 SemanticKITTI 权重后只改输出类别数。
- 分别建立隔离环境。D-FINE 作者要求 torch>=2.0.1；IS-Fusion 与 SphereFormer 为旧版 CUDA 扩展项目，需逐项解决 4090/SM89 兼容问题。
- CUDA 扩展记录 nvcc、CUDA runtime、PyTorch、GCC、SpConv/SpTr、MMCV 版本及编译架构。使用支持 SM89 的编译工具链并做实际算子 smoke test；不能只看安装成功。
- 兼容修改限制于构建、弃用 API、路径、mini infos 与单卡推理接口；任何网络结构、卷积权重布局或数学语义修改都要单列验证，不能用 strict=False/任意 reshape 混过。
- 保留 Cylinder3D 的 SpConv 1 环境。其过去直接换 SpConv 2 导致错误预测，不把环境升级算作本轮收益。
- 每个学习模型依次做严格加载、单帧、首/中/末三帧，然后 39 帧和 81 帧。记录 24 GB 实际显存峰值。先 batch=1；显存不足先修复推理保留、关闭无用训练分支，不静默减分辨率/范围/通道。

## S1：封存和补齐基线

创建 baseline_manifest.json，记录生产代码、环境、配置、输入与输出哈希、评估器版本、split、精度模式、TF32 设置及计时范围。

参考旧成绩仅用于一致性检查：
- BEVFusion 原生 mini_val：mAP=0.5790331434，NDS=0.5144708836；scene-0061：0.7117608071 / 0.6223478196。
- CenterPoint scene-0061：mAP=0.699071191，NDS=0.610700323；此前未完成 mini_val 81 帧对照，本轮补跑。
- Cylinder3D scene-0061：历史 union>0 类 mIoU=0.7199045，点准确率=0.9516034。它不是 404 帧全量成绩。
- YOLOv8s 历史 COCO AP=0.4495358；先核对完整评估文件和协议，不能只拿摘要当逐图预测。

已有同协议封存产物可复用，避免重复推理；缺少 81 帧、完整预测或同口径评估则补跑。
检测比较固定相同 GT/split/官方配置；模型保留各自原生输入预算，并在对照表列明相机尺寸、sweeps、范围和点数，结果属于整条模型路线比较。
跟踪至少固定两套检测输入：BEVFusion reference、IS-Fusion native。每一套输入分别跑 PubTracker 原始、已实现 greedy 候选、SimpleTrack；不能同时改变检测与跟踪后把全部收益归因于跟踪器。
分割为 Cylinder3D 与 SphereFormer 在同一 39/81 帧、同一原始点顺序及共同标签协议下重新评估。
对所有 GT：网络 forward、关联成本、阈值选择和出生/删除决策均不得读取评估真值。

## S2：优先运行 SimpleTrack，无训练成本

1. 先阅读作者 nuScenes 文档、MOTModel、FrameData、BBox 与 result_creation，完成现有 nuScenes 检测 JSON 到作者输入的适配。
2. 固定作者 configs/nu_configs/giou.yaml：score_threshold=0.01、max_age_since_update=2、min_hits_to_birth=0、match_type=bipartite、asso=giou、motion_model=kf、has_velo=true、nms_thres=0.1、asso_thres.giou=1.5；redundancy.mode=default。保存完整 YAML。
3. 这是作者完整 GIoU/运动/轨迹管理配置，不能只给原来的 Hungarian 代码加 KF 后改名 SimpleTrack；也不要未经实现核实就称为两阶段跟踪。
4. 使用已有 2 Hz 关键帧。10 Hz 版本需要额外中间帧检测，本轮不冒充运行；回放速度不代表输入频率或推理吞吐。
5. data_loader.pc=true 所需原始关键帧点云按作者格式提供。bbox 长宽高顺序、yaw/四元数、全局/ego/LiDAR 坐标、速度旋转和时间差逐项验证。
6. mini 适配按真实 scene/token 列表切分，首轮进程数 1，不硬编码作者全量 validation 的 150 scenes。跨场景清空状态；顺序错误、同帧重复及恢复运行需能检测。
7. 在 39 帧冻结配置，81 帧不再调参；保存完整 tracking_results.json，以 nuscenes-devkit 官方 TrackingEval 计算 AMOTA、AMOTP、IDS 等。
8. 原教学诊断仍单列：同类、score>=0.25、中心距离<2 m 的 IDSW、gap ID、FP/FN。它不能与官方 IDS 混称。
9. CPU 性能在相同检测输入、相同状态初始化下做至少 30 轮配对；报告端到端、关联及轨迹管理时间，结果 JSON 的一致性与差异案例均可回溯。

## S3：D-FINE-S 2D 检测

1. 使用作者 S/COCO 配置，原生 eval_spatial_size=[640,640]、HGNetv2 B0、三层 decoder；展开全部 include，关闭训练增强及梯度，使用作者评价分支与 checkpoint/EMA 选择方式。
2. 使用官方 COCO image_id 与稀疏 category_id。作者 80 类连续索引需正确还原，检查 bbox xywh、原图尺度、resize 逆变换。
3. 先 batch=1 跑代表性图片，再跑已有 COCO val2017 5,000 张。保留官方评估所需完整预测，显示置信度阈值不得裁掉评估输入。
4. 用 pycocotools COCOeval，保存 AP@[.5:.95]、AP50、AP75、APs/APm/APl、AR、逐类结果与原始预测。
5. 与 YOLOv8s 同一数据/评估器比较；注明各自原生预处理、阈值/NMS、输出数量、输入尺寸与权重来源。nuScenes 六路图像只有展示用途，不把 3D 框投影当作 COCO 2D 真值。
6. 原生正确后，执行作者 tools/deployment/export_onnx.py --check，用同一 checkpoint 验证 ONNX 与 PyTorch，再构建 TensorRT FP16；先核查实际 ONNX 输入/输出和 shape，不能套用旧 YOLO engine 接口。
7. TensorRT 精度通过共同 COCO 评估及几何检查才给默认开关，报告纯模型与端到端两种计时；作者 T4 延迟不是本机 4090 延迟。

## S4：IS-Fusion 融合检测

1. 使用原生六相机 + LiDAR 完整融合配置：img_scale=(384,1056)、voxel_size=[0.075,0.075,0.2]、point_cloud_range=[-54,-54,-5,54,54,3]、sweeps_num=10、5 维点特征、200 个实例/proposals。
2. 使用作者测试 pipeline、RGB 归一化和固定测试增广。保留相机内参、camera2ego、lidar2ego、lidar2camera、camera2lidar、lidar2img、img_aug_matrix、lidar_aug_matrix。
3. 不把旧 BEVFusion 的图像/体素/深度中间张量接到新模型，输入 packet 只复用原始传感器、metadata 和 token。
4. 加 hook/计数与形状记录，证明图像 backbone、LiDAR voxel/sparse 分支、场景融合、实例融合及检测 head 真实运行。输出中注明 native pipeline 的实际 sweep 数。
5. 先 scene-0061 39 帧，再 mini_val 81 帧；转换 nuScenes 格式时检查 translation/size/rotation/velocity、坐标系、单位与类别。
6. 使用官方 DetectionEval：置信度排序、同类别匹配、平面中心距离 0.5/1/2/4 m AP、TP error 与 NDS。固定 class range、max_boxes_per_sample、eval split 及官方版本。
7. 本轮各检测器采用与旧融合对照相同的 attribute 策略（旧比较为空 attribute），另存原始模型输出；不能从 GT 填属性以提升 NDS。
8. 不为可视化阈值过滤官方评估预测，不重复自己写的匹配器替代官方结果；展示用阈值单独记录。
9. 导出 per-class AP/TP errors、按距离/遮挡/点数的教学诊断及改善/退化对照图。与 BEVFusion 比较六相机+LiDAR路线，与 CenterPoint 比较保留其 LiDAR-only 标记。
10. 原生参考通过后才做图像子图后端优化。不要默认把旧 TensorRT head/LayerNorm 插件和 BEV pool CUDA 候选移植过来。

## S5：SphereFormer 点云语义分割

1. 使用作者 nuScenes spherical transformer 配置与标签映射。保留网络通道、球面/径向 window、相对位置编码及 voxel_size=[0.1,0.1,0.1]。
2. 单卡推理设 eval mode、关闭训练增强/随机 dropout、batch=1、关闭不需要的分布式入口，启用真实 validation/inference 分支。作者文件 TRAIN 中 val=false，不得直接跑训练模式。
3. 审计实际 dtype 与 SpTr 支持；先用作者可复现推理精度，不能盲目给自定义稀疏 attention 全开 FP16。
4. 输入原始关键帧 XYZ/intensity，按作者流程处理。核验 voxel inverse index，把预测恢复到每个原始点，保存 1:1 的点标签；抽样或 voxel_max 裁剪不能丢失评估点。
5. 从作者 0–15 网络类别映射到项目统一 1–16 语义类别；原始 32 类 GT 走官方映射，noise/ignore 不计入有效点。列出映射表并做逐类检查。
6. 先 39 帧，再 81 帧，与 Cylinder3D 共用评估器。保存 confusion matrix、16 类 IoU、每类 GT/预测/union 数、valid-class 标记、union>0 的 mIoU 及有效点准确率；没有出现的类记 N/A，不伪填 0 或称全量官方成绩。
7. 两模型具备相同资源后，补跑现存 404 帧的语义对照，作为 mini 全集辅助分析；单列 39、81、404 范围，明确预训练数据可能与 mini 重叠，不据此宣称独立泛化。
8. 展示预测、GT、误差点与远距离稀疏目标切片；颜色统一但保留各模型标签文件。不得把点类别外推到不可观测体素充当 occupancy GT。

## S6：流程与后端优化，逐项做 A/B

旧 BEV pooling pack4 在三组真实输入均慢于 upstream，已无稳定端到端收益；本轮保留 upstream，不继续无证据扫 block 大小。
旧 Sparse4D 平均约 80.2% 的端到端时间用于预处理、8.2% 用于标准化；这是流程优化的线索，必须先测新模型自己的 profile。

按累计成本排序，每次只改变一个因素：
1. 数据准备：常驻模型；限制长度的 CPU 预取队列；独立记录磁盘读取/图像解码/resize/归一化、点云读取和坐标变换。
2. 静态与帧级 metadata：预解析 token/路径及标定矩阵；帧绑定的 ego_pose、timestamp 与 sweep transform 按实际索引读取，不能跨帧误复用。
3. 坐标变换：同一帧矩阵构造一次，批量转换 bbox/角点/速度/六相机投影，减少逐框 Python/Quaternion；平移不得施加于速度。保持作者坐标和精度语义。
4. 内存传输：预分配/复用缓冲区、固定大小 pinned memory、实测有收益的 non_blocking H2D。控制队列显存与 RAM；不要在预测仍引用时覆盖 buffer。
5. D-FINE 原生 ONNX/TRT FP16 优先；IS-Fusion 先尝试兼容的图像 backbone/neck 子图。geometry/index、稀疏映射与敏感 head 保留参考精度，记录边界 casts。
6. Torch compile/CUDA Graph/自定义 CUDA 只在该模型有可复现热点且兼容时增设独立候选，不作为四模块都完成的前置条件。

计时要求：
- 同权重、同样本、同 shape/语义，预热后至少 30 轮交错配对，报告 P50/P95、均值/波动与绝对毫秒。
- 分开 cold start、数据准备、H2D、GPU forward、decode/D2H、标准化/坐标转换、可视化；GPU 计时用正确同步的 CUDA event，端到端用 wall clock。
- 原始数据加载到可消费预测作为端到端主口径，双方同缓存条件。不得预存图像处理结果后与原路径含磁盘 I/O 的时间直接比。
- 官方精度评估使用完整预测。对同模型后端优化，3D mAP/NDS、2D AP、分割 mIoU 的绝对下降各不超过 0.005 是本轮工程预算；这是项目阈值，不是官方认证标准。
- 同时检查原图 bbox 还原、3D 几何/index、点覆盖、非有限值和中间数值误差。旧 FP16 曾只通过最终 AP/NDS 而未通过全部诊断，不能重复只看两个总分。
- float 舍入容差须依据参考重复噪声和任务几何单位定义。离散 voxel/index、类别映射、token/point 顺序必须严格正确。
- 只有稳定提速、精度及几何检查均通过的候选才可采用。否则保留 native 并记录失败；不同模型精度/速度比较与同模型后端优化分开报告。

## S7：编排、可视化与产物

实现以下新增入口，参数可按仓库惯例完善，但 README 的复现命令必须实际可执行：
- configs/data_matched_optimization.yaml：版本/权重/数据/split、原生与候选、逐项开关、验收预算。
- tools/run_data_matched_optimization.py：preflight、baseline_eval、simpletrack、dfine、isfusion、sphereformer、pipeline_opt、image_backend、visualize、summarize；阶段可独立运行/恢复，all 只包含本轮已定义的任务。
- 一次示意调用为：
~~~bash
python perception_lab/tools/run_data_matched_optimization.py --phase preflight
python perception_lab/tools/run_data_matched_optimization.py --phase all
~~~
以上为待实现接口；在报告中给出实际环境启动器和最终命令。

复用现有 9094/Rerun 回放设施，增加模型选择和成对对照。首/中/末帧核对：
- 相同 sample token 下六相机、点云、检测框、track ID/轨迹和点级语义；同 ID 跨视图同色。
- GT/误差作为可切换显示层，模型输入不包含 GT。模型缺失时显示真实状态，不使用前一帧假填。
- nuScenes 上 D-FINE 2D 框只可视化；SphereFormer 点语义不得显示为已评估占用体素。
- 冻结相同帧选择，保存代表性改善、退化与失败图片及一个短回放片段；避免只挑更好看的帧。

在 reports/data_matched_optimization_20261012/ 生成：
- report.md：本轮真实完成范围、最终采用/保留决定、数据/配置/版本/权重来源、表格与可复现命令。
- status.json：每阶段真实状态、开始/结束时间、阻塞理由、采用与否。
- metrics.csv、metrics.json：baseline/candidate、task、split、frame/image/point counts、official/diagnostic protocol、mAP/NDS/AMOTA/AMOTP/IDS/mIoU/AP、P50/P95/显存；不适用字段留空。
- baseline_manifest.json、preflight.json、resource_manifest.json、environment.md。
- official_eval/、confusion/、profile/、badcases/、visualization/：保存官方摘要、运行命令、输入/输出哈希和代表性证据；完整大预测文件可本地保存，仓库放其摘要/哈希/生成方式。
- 保留已经发布的 plan.md。新 report.md 不复制计划目标当成绩。

重要失败样本必须分类：远距/遮挡/小目标、校准/坐标/时间错位、原生模型误检漏检、关联/轨迹管理、点云逆映射、CUDA/权重/资源失败。
分析先给可观测证据，推测标为待验证，给能区分原因的下一项实验。

## S8：验收、提交与发布

完成必要的适配测试：坐标/速度与 bbox 轴顺序、类别映射、点逆映射、跨场景 reset、置信度输出完整性、作者结果 parity。不要写只检查实现细节的形式测试。
运行仓库与改动相关的检查，以及实际单帧、39/81 帧、官方评估、首/中/末可视化；COCO/404 帧用实际完成数报告。
更新 README 与 TASK_STATE，清楚写 executed 或 blocked；本轮发布计划不等于实验执行。
按 AGENTS.md 的既有授权提交必要代码/配置和可分享结果，推送并确认远端 main/所用结果分支 SHA、报告和指标可读取。
若推送失败，保留本地提交并记录错误，不声称远端完成。最终向 BOSS 汇报：采用了什么、测了多少、增益/退化、失败原因、保留哪些基线、链接与提交。

完成标准：四模块各有可核对的真实结果或具体阻塞证据；所有采用决定来自共同数据/协议实测。允许 partial，禁止把“模型更晚发表”当作性能改进的证据。
