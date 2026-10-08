# Instruction 20261008_01：单卡 RTX 4090 感知模型部署、优化与可视化

## 执行目标

Codex：在 BOSS 的 CARperception 工作站实际实现并运行本指令。完成 **BEVFusion 相机–LiDAR 融合 → TensorRT FP16 与 BEV pooling 算子优化 → Sparse4D v3 时序检测/跟踪 → GaussianWorld 流式占用预测**，接入现有结果格式、Rerun 回放和失败案例分析，提交可复现代码、真实实验结果及报告。按阶段连续推进，不以计划、接口占位、下载成功或上游 Demo 截图代替模型部署。

硬件目标为单张 RTX 4090 / 24GB。模型依次运行，第一轮使用官方预训练权重完成推理、评估与优化；训练/微调、VLA 和车端硬件移植作为后续独立实验。4090 的实测时延只能标为桌面 GPU 结果。

先阅读根目录 `AGENTS.md`、`README.md`、`perception_lab/TASK_STATE.md`，以及以下文件：

- `perception_lab/reports/industry_deployment_review_20261008/report.md`，尤其第 9 节的硬件与配置核对。
- `perception_lab/reports/pipeline_timing/report.md`、`reports/official_detection_mini/report.md`。
- `perception_lab/tools/prepare_mini.py`、`geometry.py`、`lidar_smoke.py`、`export_nuscenes_detection.py`、`rerun_mini.py` 和现有失败工作台实现。

派发时远端 main 为 `a77a08b774d0f70a07a8d361cfb69710ca0813ec`。执行时先获取最新 main，并在报告记录实际起始 commit。本指令扩展现有 mini 教学闭环；历史模型、跟踪候选、评估报告及默认回放保留，新增结果使用独立模型命名空间。不要启动旧规划中的 `--phase all`。

## 阶段与验收

| 阶段 | 工作 | 最低可验收成果 |
|---|---|---|
| S0 / preflight | 硬件、数据、依赖与输入契约 | GPU/版本清单、固定 sample 列表、独立环境方案、实际可用数据范围 |
| S1 / bevfusion_reference | BEVFusion R50 参考推理 | 本项目 scene-0061 全 39 帧真实六相机＋LiDAR 预测、坐标审计、官方代码检测评估 |
| S2 / bevfusion_tensorrt | CUDA-BEVFusion / TensorRT FP16 | 同权重、同输入的实际 engine、39 帧输出、数值/指标差异、预热时延与显存 |
| S3 / bev_pooling | BEV pooling CUDA 优化 | 上游参考和自主修改的候选实现、真实形状正确性对照、配对计时及整模型效果 |
| S4 / sparse4d | Sparse4D v3 R50 | 39 帧时序检测与 ID、场景切换重置验证、mini_val 检测/跟踪评估 |
| S5 / gaussianworld | GaussianWorld 流式占用 | 原配置显存验收；可运行时生成连续占用结果、语义图例与时序回放；标签可用时评估 |
| S6 / visualize | 同步展示与案例分析 | 六相机、BEV、3D、轨迹/占用统一时间轴；实际打开验证；每模块代表性问题案例 |
| S7 / summarize | 汇总、复现与发布 | 报告、状态、指标、产物清单、运行命令、代码/配置及远端发布核对 |

每完成一个模型，立即保存结果、生成该模块回放和阶段报告，再继续下一模型。S2/S3 依赖 BEVFusion 的正确参考；Sparse4D 与 GaussianWorld 可独立推进。遇到阻塞先定位并修复可控的代码、配置和依赖问题，保留失败日志；无法解决时记录具体缺项，继续独立阶段，最终报告全部实际状态。

## S0：准备本机环境和固定实验输入

1. 检查本地未提交改动，避免覆盖 BOSS 的工作。基于最新 main 建立工作分支 `codex/20261008_4090_perception_deployment`；必要时使用独立 worktree。保留当前可运行环境，为 BEVFusion、TensorRT、Sparse4D、GaussianWorld 分别建立隔离环境或容器。
2. 记录 `nvidia-smi`、GPU 名称/总显存/运行前可用显存、驱动、CUDA runtime/toolkit、cuDNN、TensorRT、Python、PyTorch、MMCV/MMDetection3D、编译器和自定义扩展版本。核对 SM 8.9 编译支持。模型运行期间不终止无关进程、不全局升级现有依赖。
3. 复用 `perception_lab/data/nuscenes/` 中的 mini。从官方元数据重新生成 **scene-0061 完整 39 帧**列表，固定 `scene_token/sample_token/timestamp_us`、六相机顺序、点云及实际 sweep 列表，不直接信任历史绝对路径。保存输入文件与元数据哈希。
4. 另从官方 `create_splits_scenes()` 的 `mini_val` 得到全部场景/帧，作为多场景评估范围。适配器先打通 scene-0061，再跑 mini_val。不要手写猜测场景名或把 39 帧称为完整 val。mini 与预训练数据可能重叠，报告注明这一限制。
5. 必要的公开权重、源码、mini 扩展按官方来源获取，记录上游 **commit**、配置、权重 URL/大小/SHA256、环境锁定和兼容补丁；文件 blob SHA 不能当作仓库 commit。大型数据、权重、engine、环境及第三方源码按现有规则留在本地。完整 trainval val 已具备时可追加评估；其下载和原始多卡训练不作为本轮前置任务。

### 统一输入输出契约

- 复用现有帧数据与几何工具；模型适配器核对相机顺序、图像缩放/裁剪/归一化、内参更新、相机各自的 calibrated_sensor/ego_pose、LiDAR 与相机时间对齐、历史 sweep 到当前帧变换、点云强度和时间字段。推理用确定的测试预处理，记录模型实际使用的传感器与历史帧。
- 3D 检测输出复用当前标准化字段：`sample_token`、`scene_token`、`timestamp_us`、`source=measured`、`boxes3d`；框含 `center_xyz`、`size_wlh`、`rotation_wxyz`、`velocity_xy`、类别与模型分数。标准化到 **当前 LIDAR_TOP 对应 ego 系、gravity centre、米、米/秒**。逐模型核对原生坐标系、尺寸顺序、yaw、bottom/gravity centre，不重复变换。
- 每个真实执行帧均有记录，真实空输出为 `[]`，缺失帧须修复或标记缺失；预测不能从 GT、历史其他模型输出或下载的预测文件填补。真值仅进入评估/核验/独立展示层。
- 跟踪保存模型原生 ID、分数、时间戳、类别和轨迹来源，跨场景独立 ID 空间；占用保存预测类别网格、坐标范围/体素大小/轴序/类别表/empty 与 ignore 定义。Gaussian 是中间表示，语义占用网格是该任务的验收输出。
- 报告或 manifest 记录原始输出、标准化输出、实际配置和权重哈希。渲染阈值与评估输入分开，禁止为了画面干净提前删掉低分预测。

## S1：BEVFusion R50 六相机＋LiDAR 参考推理

采用 NVIDIA CUDA-BEVFusion 提供的 **ResNet50 BEVFusion-Base、六相机 256×704、batch=1** 作为参考与 TensorRT 的共同模型。优先使用 NVIDIA 发布包中的 R50 PyTorch 权重与配套配置；核对相机、LiDAR 分支和融合层均真实执行。

上游入口：

- [NVIDIA CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/README.md)
- [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion)，NVIDIA 指定的兼容 commit 为 `db75150717a9462cb60241e36ba28d65f6908607`。

1. 阅读对应 commit 安装与推理入口，按其依赖建立独立环境，严格加载权重，核对缺失/多余 key 与张量形状。保留官方稀疏卷积布局；不得随意 reshape 权重或静默跳过层。
2. 官方 example-data 可以验证安装/导出，但验收必须切换到本项目 mini 数据。实现连续帧适配与常驻模型 runner，先一帧验证，再运行完整 39 帧，随后覆盖 mini_val；不能每帧重新启动 Python 或重复加载模型。
3. 检查至少三帧的六相机投影与 BEV/3D 框，保存原生/标准化框往返审计。不要用“能画出框”代替坐标正确性验证。
4. 调用仓库已有官方检测导出器与 nuScenes `DetectionEval/detection_cvpr_2019`，扩展传感器 meta 为真实的 `use_camera=true/use_lidar=true`，并处理模型实际属性来源。复用已验证 ego→global 变换，不改官方匹配、排序、距离阈值或 AP 代码。
5. 输出 scene-0061 自定义 split 与 mini_val 各自的原始预测、官方 `metrics_summary.json/metrics_details.json`、每类 AP、mAP/NDS、代表帧和失败案例。没有真实属性预测时空属性及其对 NDS 的影响需单独说明；任何规则推导属性须标明规则，禁止用 GT 补属性。

## S2：CUDA-BEVFusion / TensorRT FP16 部署

1. 固定与 S1 **完全对应**的 R50 权重、网络和预处理。先核对官方发布 ONNX 与 PyTorch 权重的对应关系；来源不明确时从参考权重重新导出。记录所有 ONNX/engine 哈希、输入输出 shape、动态点数 profile、插件版本、构建配置及 workspace 上限。
2. 根据实际版本选择可兼容的 CUDA、TensorRT 与 SpConv 后端。NVIDIA 示例测于 TensorRT 8.6；不要假设旧 API/插件在 TensorRT 新主版本下直接可用。优先固定兼容版本，有补丁就提交补丁与编译日志。engine 在本机 4090 构建，保留具体 GPU/版本信息。
3. 阅读上游 `tool/environment.sh`、`tool/build_trt_engine.sh`、`src/onnx/make_pb.sh`、`tool/run.sh`、`tool/pybev.py` 后执行对应入口；首轮选择 R50 + FP16。官方原始命令需要实际配置路径：

```bash
# 在固定 commit 的 CUDA-BEVFusion 目录，完成 environment.sh 配置后执行。
source tool/environment.sh
bash tool/build_trt_engine.sh
bash src/onnx/make_pb.sh
bash tool/run.sh
```

4. 将本项目逐帧输入送入真实 CUDA/TensorRT 实现，核对它与 S1 的图像处理、点云/sweeps、几何矩阵、框解码/NMS/速度/分数一致。固定相机 shape，同时测试 mini 中真实点数的最小/中位/最大输入及 profile 边界，不能只跑示例一帧。
5. 对完全相同 39 帧及 mini_val 做参考/TensorRT 配对比较：能取得时比较关键中间张量及 NMS 前输出，报告 shape、NaN/Inf、绝对/相对误差统计；再比较最终框、分类/分数、匹配框中心/尺寸/yaw/速度和官方评估指标差值。浮点误差可能引发 NMS 边界变化，需解释差异，不把框列表哈希不同直接等同部署失败。
6. 在看到优化结果前固定数值容差和验收预算，并写入配置。默认 FP16 指标预算为同一 split 的 mAP、NDS 各下降不超过 **0.005（0–1 标度，即 0.5 个百分点）**；这是本项目的工程验收目标，不是官方标准。逐类 AP 和几何偏差仍须复核，未达目标则定位预处理、精度或插件问题，不能事后放宽预算后写“无损”。

首轮 FP16 通过后，INT8/PTQ 才作为加分实验：使用与评估样本隔离的实际校准集，记录校准范围/样本/缓存哈希及精度变化。缺少有效校准数据时保留 FP16 成果，不借用官方 FPS 当本机实测。

## S3：BEV pooling CUDA 算子优化

本阶段要交付 **可定位的代码修改与测量结果**。仅编译调用上游 CUDA kernel，应写作“部署上游算子”，不写成自主算子开发。

1. 从 S1/S2 捕获真实 BEV pooling 的输入、索引、区间、dtype、shape 和输出，将上游实现作为参考。先用 PyTorch profiler/Nsight Systems 分离 backbone、view transform、pooling、稀疏卷积、融合、head、H2D/D2H 及同步；用 Nsight Compute 分析 pooling 的内存访问、占用率和实际瓶颈。
2. 根据真实瓶颈实现至少一个可切换 CUDA 候选：例如读写合并、半精度向量化、线程/warp 工作分配、重复索引读取减少或中间张量消除。参考 [NVIDIA BEV pooling 技术说明](https://developer.nvidia.com/blog/accelerating-bev-pooling-on-nvidia-gpus-for-physical-ai-applications)，但实际实现须匹配本项目算法与 TensorRT 插件版本。禁止仅调计时边界获得“提速”。
3. 先验证真实帧和有意义边界输入：空有效投影、单点、多个贡献到同一 cell、边界索引、真实最小/最大形状。保持参考累加/索引语义；FP16 结果用预先固定的容差报告最大与分位数误差，检查越界和非有限值。发生精度或内存错误先修复。
4. 对参考/候选做至少 **30 轮配对测量**，交替顺序、相同输入、独立预热、CUDA 同步边界，记录逐轮数据与分布；同时在 S2 完整分支上比较时延、峰值显存和检测指标。报告算子加速比、整模型加速比与绝对毫秒数，不把 kernel 加速比当成整系统加速比。
5. 有稳定收益且精度通过才选为默认部署候选；无收益也保留真实代码、测量和原因，保持参考实现默认。本阶段“完成实验”与“获得提速”分别标注，不能预填收益。

## S4：Sparse4D v3 R50 时序检测与跟踪

使用 [Sparse4D 官方仓库](https://github.com/HorizonRobotics/Sparse4D)、配置 `projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py` 和 [官方 R50 v3 权重](https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth)。第一轮六相机 256×704、batch=1、单 GPU 推理，按作者定义保留实例记忆。

1. 依作者 `docs/quick_start.md` 安装并编译 deformable aggregation CUDA 扩展，锁定环境/上游 commit，严格加载官方权重。核对 `nuscenes_kmeans900.npy` 等先验资产来源与哈希，不在 mini_val 上重新拟合先验后冒充官方配置。
2. 将官方数据转换器及时间信息适配为 `v1.0-mini`。所选配置推理输入为六相机；训练代码中的 LiDAR 深度监督不等于相机–LiDAR 融合推理。meta 据实际输入填写。
3. 单 scene 按时间戳递增处理，保留历史实例/feature/anchor、自车运动变换与时间间隔；场景切换必须重置。验证整段运行与“保存状态→恢复继续”的一致性或明确框架不支持恢复；不得每帧清空历史后声称时序已实现。
4. 保存 scene-0061 全 39 帧原生检测和跟踪 ID，并跑完整 mini_val 的检测/跟踪输出。使用官方 `DetectionEval` 和 `TrackingEval` 分别计算 mAP/NDS 与 AMOTA/AMOTP/IDS；类别按官方检测 10 类/跟踪 7 类转换，保留完整 split token 覆盖与场景 ID 隔离。
5. 现有工作台的固定阈值 FN/FP/ID 诊断可用作案例复核，标明其匹配口径；不能与官方 TrackingEval 的 IDS 混为同一数值。只有在相同 split、同一评估协议上重跑 CenterPoint/PubTracker，才做两条路线的数值对照，禁止拿 scene-0061 旧指标对比新 mini_val。

## S5：GaussianWorld 原配置资源验收与占用预测

使用 [GaussianWorld 作者仓库](https://github.com/zuosc19/GaussianWorld)、`config/nusc_surroundocc_stream_eval.py` 及 README 中 **GaussianWorld streaming** 权重。原始配置为 R101-DCN、六相机 864×1600、25,600 Gaussian、200×200×16 网格、batch=1、`amp=False`；数据与类别按作者 SurroundOcc 协议。

1. 按作者 `docs/installation.md` 单独建立环境，编译 Gaussian encoder 与 local aggregation 扩展；保留严格权重加载、实际 grid/range/类别映射核验。官方 `scripts/eval_stream.sh`/`eval.py` 可作为参考，不随意替换为空间检测或点云分割模型。
2. 先用本项目一帧，再连续短段测量 allocated/reserved 峰值和进程 GPU 显存；确认原结构在 24GB 上的实际资源需求后扩展到 39 帧。原作者评估脚本会先搬运整个 clip，优先检查输入片段驻留、临时张量、无梯度推理等开销。
3. 如 OOM，优先改为逐帧 CPU→GPU 搬运，保持历史 anchor、几何/时间信息、原始网络、Gaussian 数量和输出网格；对改动前后能运行的相同短段比较输出。混合精度需先检查自定义 CUDA 支持及误差，不能直接全局开启后写“无损”。降低分辨率/减少 Gaussian/改 voxel grid 是另一配置实验，不能覆盖原配置验收或假定与原权重等价。
4. 按模型时间顺序传播状态并记录场景/clip 边界。官方评估保持作者 clip/重置协议；39 帧连续回放若使用全 scene 持续状态，另记为 scene-stream 协议，避免混用指标。
5. 对缺失 SurroundOcc 真值的 mini 帧，可实现**真实无标签推理路径**：只调整数据载入和评估分支，将 label 从预测路径移除，核对其不会改变预测计算；不创建伪标签，也不把 LiDARSeg 当体素真值。有匹配 token 的官方占用标签时，按作者 ignore/empty 定义计算 IoU/mIoU/逐类 IoU，报告有效标签覆盖。缺标签仍交付推理和可视化，评估标为 blocked_data。
6. 保存逐帧语义占用网格、几何定义、类别图例和状态来源，接入 Rerun 点/体素展示、BEV 语义切片及六相机上下文。模型在适配后仍 OOM 时，提交实际峰值、失败位置和已尝试的输出保持措施，S5 标记 blocked_resource；其他已完成模型照常发布。

## 统一计时、可视化与报告要求

### 性能测量

- 模型在评估模式、无梯度下常驻，先完成独立预热，再测全部目标帧。无状态模型可单独报告同帧重复计时；时序模型按原始顺序处理并保持正确状态，不把同帧连续喂入的状态污染当作有效性能测量。
- 报告冷启动（导入/权重加载/engine 构建）、数据读取/预处理、H2D、模型、后处理、D2H、渲染/视频各自边界。GPU 耗时使用 CUDA event 或正确同步的墙钟；端到端墙钟包括真实输入到标准化输出，不把离线视频生成塞进模型 forward。
- 对真实逐帧时延提供样本数、均值、P50/P95、首帧、总耗时、峰值显存和输入点数/图像 shape。非 PyTorch 后端同时记录进程 GPU 显存；不能只拿 PyTorch allocator 数字表示 TensorRT 总显存。FPS 由明确范围的实测时延计算。
- 全流程先按帧或按模型顺序完成；如果比较各阶段之和，写明为分阶段测量，不声称所有模型同时常驻或达到车端实时闭环。

### Rerun、视频与失败案例

1. 在现有时间轴上增加独立实体路径，如 `models/bevfusion_reference`、`bevfusion_trt`、`sparse4d`、`gaussianworld`，采用一致 `sample_token/frame_id/timestamp` 和标准化坐标。六相机、点云、检测、ID/轨迹、占用逐帧清理并同步；GT 独立开关，预测图层不混入真值。
2. BEVFusion 展示相机与点云融合后的框；Sparse4D 展示稳定 ID 与轨迹；GaussianWorld 展示带类别图例的语义占用。通过既有 Rerun/案例浏览器实际打开，核对首/中/末帧、跨相机投影、场景切换、轨迹清理与占用轴方向，保存验证截图。
3. 每个可运行模块输出 scene-0061 同步 MP4、至少首/中/末三个代表帧，以及最多 5 个有证据的失败/异常案例。案例写清 token、类别、距离、分数/无分数、现象、可能原因、核验依据与下一步。根因未验证时标“假设”，不能凭画面宣称因果。
4. 用实际解码检查视频帧数与覆盖；2fps 等展示速度是播放速度，与推理 FPS 分开记录。完整 RRD/网格/视频过大时按现有规则留本地，提交代表图、紧凑摘要、哈希和生成命令。

### 代码入口、产物与发布

新增统一配置 `perception_lab/configs/deployment_4090.yaml` 和编排入口 `perception_lab/tools/run_4090_deployment.py`，调用隔离环境内的真实模型适配器，不强求全部依赖安装到同一个 Python。**下列是本任务需要实现并实际验证的新 CLI 契约，派发时脚本尚不存在：**

```bash
cd perception_lab
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage preflight
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage bevfusion_reference
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage bevfusion_tensorrt
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage bev_pooling
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage sparse4d
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage gaussianworld
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage visualize
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage summarize
```

配置包含本机路径、环境 Python、模型/权重/上游版本、split、sample 列表、预处理、精度、shape profile、数值容差及计时设置。支持断点续跑：只复用配置/权重/输入哈希完全匹配的产物；已运行的可复用结果标明来源，未运行阶段不能生成假成功状态。所有命令、工作目录、环境和返回码留痕。

代码、兼容补丁及针对性验证提交到仓库现有 tools/configs/tests；优先验证跨模块坐标、格式、时序状态、CUDA 正确性与导出一致性，不写只验证文档/占位返回值的测试。旧模型的针对性测试保持通过。

统一实验目录：`perception_lab/reports/deployment_4090_20261008/`。至少发布：

- `report.md`：分阶段实测、模型/数据范围、数值一致性、精度、时延/显存、失败案例、未完成项及来源。
- `status.json`：每阶段/子任务的 `not_started/running/passed/partial/blocked/failed`、原因、run ID、数据覆盖、真实产物路径/哈希；推理通过而标签缺失时分别记录推理与评估状态。
- `metrics.csv`：模型、后端、精度、split、帧数、评估协议、mAP/NDS/AMOTA/AMOTP/IDS/占用 IoU、时延 P50/P95、显存和相对参考差值。未测数值留空并注明原因，零不能用来代替未知。
- `artifacts_manifest.json`：上游 commit、配置/权重/输入/预测/ONNX/engine 哈希、环境、产物与本地大型文件位置。
- `run_commands.md`、代表图、失败案例摘要和必要的官方评估 JSON；原始逐帧计时与配对算子数据保存为 CSV。

每阶段更新本目录状态和报告；完成后更新 README 的报告入口及 `perception_lab/TASK_STATE.md` 的实际状态。按 AGENTS.md 提交并推送实现与结果，将验收通过的工作合入 main，并核对远端文件、提交号和报告内容。未完成项保持可追溯，不把发布指令写成实验已完成。

## 最终交付判断

本轮完整部署闭环要求：BEVFusion 真实融合参考、TensorRT FP16 一致性、BEV pooling 候选实验、Sparse4D 时序检测/跟踪、GaussianWorld 占用预测、同步展示、实测报告和远端发布均有证据。算子无提速须如实报告；占用标签缺失只影响对应精度评估；模型无法在 24GB 运行时标记资源阻塞，整体保留 partial，不能宣称全流程全部通过。

最终向 BOSS 提交具体 commit、报告链接、可直接运行的命令、实际精度/时延/显存和剩余问题。优先让 BOSS 能立即打开新增模块的真实结果，再依据失败案例决定后续微调或车端部署。
