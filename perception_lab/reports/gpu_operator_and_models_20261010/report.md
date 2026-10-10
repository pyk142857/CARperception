# 自动驾驶感知模型原理与 GPU 算子优化

日期：2026-10-10。适用于 CARperception 单卡 RTX 4090 的部署学习和工程优化。

GPU 优化应从实际关键路径出发：确定哪段时间占比大，再判断它受计算、访存、同步还是启动开销限制，最后选择保留模型计算语义的实现变化。参数量、FLOPs、显存占用和 nvidia-smi 利用率都不能单独决定哪个算子值得优化。

本文结合论文与官方源码讲解 BEVFusion、Sparse4D v3、GaussianWorld，并对照 PointPillars、CenterPoint、Cylinder3D、BEVFormer、MapTR、SurroundOcc 及项目中的图像模型。BEVFusion 是 ICRA 2023 的 MIT 版本，Sparse4D v3 是 2023 年预印本，GaussianWorld 是 CVPR 2025；“新增部署”描述项目范围，不等于这些模型刚在 2026 年提出。[7–9]

本文是原理和源码分析，没有新增 GPU 性能实验。候选优化的收益须由本机 profiler 与配对实验验证。核对时项目 main 尚未发布 deployment_4090_20261008 实验报告；已有性能证据来自 2026-09-30 的 pipeline_timing。[21]

**一、先区分模型、算子和调度**

模型定义任务与计算结构，例如“六相机和 LiDAR 如何融合得到 3D 框”。算子是其中一项计算，例如矩阵乘法、卷积、双线性采样、分段求和。CUDA kernel 是执行算子的 GPU 程序；一个算子可能调用多个 kernel，一个融合 kernel 也可以执行多个算子。TensorRT 插件将自定义算子接入执行图，插件本身不保证 kernel 更快。

warp 是以 32 个线程为一组的执行单位；block 是共享资源并可进行组内同步的线程集合；SM 是执行线程与指令的处理单元。occupancy 表示活跃 warp 相对于硬件上限的比例。提高 occupancy 有时能隐藏访存等待，但也可能导致寄存器溢出或牺牲数据复用，不能把最高 occupancy 当成优化目标。[1–2]

| 类型 | 示例 | 验收方式 |
|---|---|---|
| 系统与调度优化 | 模型常驻、去掉无用同步、输入搬运与计算重叠 | 端到端时延、吞吐、输出一致性 |
| 算子实现优化 | 更少访存、重排索引、融合计算、修改线程分工 | 算子正确性、kernel 时延、整模型收益 |
| 精度优化 | FP16 输入、FP32 累加、INT8 校准 | 数值误差、同协议任务指标、时延与显存 |
| 算法结构改变 | 更少 Gaussian、更粗 voxel、不同 backbone、删历史帧 | 作为新配置评估，通常涉及重新训练或权重适配 |

同样是“变快”，后三项的精度条件和工程代价不同。尤其不能把改变任务分辨率所得的速度，称为原模型的无损算子优化。

**二、怎样判断优化空间**

首先定义时间边界：原始传感器数据到标准化预测的端到端时间、模型 forward、预处理、后处理、渲染分别测量。CUDA 异步执行，普通 Python 计时可能只测到任务提交；单算子使用 CUDA event，端到端测量在正确边界等待 GPU 完成。模型先常驻并预热；时序模型保持真实时间顺序与状态。[3–5]

工具按层次使用：PyTorch Profiler 定位框架算子与内存；Nsight Systems 查看 CPU、CUDA、拷贝、同步和空闲间隔；Nsight Compute 对少数热点 kernel 分析带宽、缓存、指令、寄存器、warp stall 和 occupancy。深度 profiler 会改变执行条件，最终速度在不开 profiler 的独立运行中确认。[2–5]

| 观测 | 可能瓶颈 | 值得验证的改法 |
|---|---|---|
| GPU 执行间隙大，CPU 忙于启动、读图或后处理 | 系统供给/同步 | 常驻、预取、批量提交、降低同步频率 |
| kernel 大量读取，DRAM/L2 接近对应带宽上限 | 带宽 | 消除中间张量、改善连续访问、复用数据 |
| 随机 gather、带宽不高且等待多 | 访存延迟 | 增加可并行工作、改善布局和局部性 |
| 大量写入集中到同一输出地址 | 原子操作竞争 | 按输出分组，局部 reduction 后少量写出 |
| 密集矩阵/卷积占主导且尺寸合适 | 计算 | 使用成熟 Tensor Core 路径，改善 layout/融合 |
| 许多极短 kernel，总体启动开销大 | 调度 | 融合或满足条件的 CUDA Graph |
| 长短工作混杂、少量线程拖尾 | 工作量不均 | 按区间长度分组、warp/block 协作 |
| allocated 很大，存在重复拷贝/超长驻留 | 内存生命周期 | 逐帧搬运、复用 buffer、提前释放中间结果 |

“带宽不高”不能推出“计算受限”：随机访问、低并发、指令依赖也会让计算和带宽都没跑满。Roofline 必须与实际指令类型、缓存层次和 stall 结合解释。[2]

设一个 kernel 执行 F 次浮点操作、发生 B 字节的实际显存流量，算术强度为：

\[
I=F/B,\qquad P_{\mathrm{roof}}=\min(P_{\mathrm{compute}},BW\cdot I).
\]

相应粗略时间下界为 \(\max(F/P_{\mathrm{compute}},B/BW)\)，不包含所有启动、同步和依赖成本。带宽受限且接近上界时，需要减少字节或增加复用；计算受限且接近上界时，继续调 block 大小通常空间有限。低于上界只能提示有差距，不能保证差距全部可消除。[2]

再用 Amdahl 定律判断整模型价值。若热点占关键路径的比例为 p、能加速 s 倍，其余成本不变：

\[
S_{\mathrm{total}}=\frac{1}{(1-p)+p/s}.
\]

教学例子：总时延 80 ms，其中热点 30 ms，将热点变为 10 ms，总时延变为 60 ms，整模型加速 1.33 倍；即使完全消除热点，最多也只有 1.60 倍。若某热点仅占 5%，消除它最多带来约 1.053 倍。存在并行重叠时，p 应按关键路径判断，不能直接把所有 kernel 时间相加当总时延。

决定是否投入时同时看：绝对毫秒数、可消除的成本、修改后是否保持语义、真实 shape 分布、实现/验证成本。参数少不等于快，稀疏不等于快，GPU 利用率高不等于计算单元跑满，显存省不等于延迟一定下降。

**三、优化可以从哪些方面入手**

1. **减少中间结果。** 将“乘法生成张量→排序/搬运→归约”改为在归约时按需读原始输入并相乘，避免写回大张量。这往往比把同一条乘法指令换成另一条指令更有价值。
2. **改善 layout 和访问。** 相邻线程尽量读取连续地址；通道维连续时采用合适的向量化加载；避免反复 permute 后 contiguous。AoS 把每个元素的字段打包，SoA 把同类字段放成连续数组，哪种更合适取决于线程访问方式。
3. **复用索引与不变量。** 可固定的映射在推理前计算；多个通道共享的 depth、索引和坐标读取不必每通道重复。但内外参、图像增强、网格或参考时间变化后不能盲目复用映射。
4. **减少冲突与拖尾。** 为每个输出分配 owner，在寄存器/共享内存中归约后写出；长区间由 warp/block 合作，短区间维持较小粒度。并行归约改变浮点加法顺序，需要误差验证。
5. **控制寄存器和共享内存。** 展开循环和加大通道 tile 可以减少指令或读取，却可能增加寄存器压力与 spill。共享内存只在存在可利用复用时有意义，搬进去也有成本。
6. **选择适合的精度。** FP16/INT8 能减少数据量或利用硬件计算单元，但 scatter/gather 不是自动变成 Tensor Core 计算。低精度累加与低精度存储分别评估；FP16 输入配 FP32 累加是常见的折中。量化要核对校准集与任务精度。[1,6]
7. **减少调度和同步。** 预分配 buffer、减少 cudaMalloc、CPU .item() 和 D2H 小标量；满足 shape、地址、执行图和状态约束时再用 CUDA Graph。图捕获不会自动解决算法中的动态数据依赖。
8. **先用成熟库。** 对标准卷积和 GEMM，优先使用 cuDNN、cuBLAS、TensorRT 或框架优化路径；自定义 CUDA 更适合有明确瓶颈的不规则采样、归约和特定融合。训练反向与推理前向的热点不一定相同。

实现工具按计算形态选择：规则张量计算和融合可先用 PyTorch 编译路径或 Triton 做原型；不规则索引、动态覆盖列表、复杂分段归约与精细线程协作可考虑 CUDA C++，并复用 CUB 等成熟基础组件。TensorRT 负责图级融合、执行与精度策略，自定义插件接入特殊算子。任何后端选择都需要与已有实现实测比较，换语言或框架不构成收益证据。

**四、模型的共同关系**

| 模型 | 主要输入 | 核心表示 | 主要任务 |
|---|---|---|---|
| PointPillars | LiDAR | pillar 特征→BEV 伪图像 | 3D 检测 |
| CenterPoint | LiDAR | voxel/pillar 编码→BEV 中心热图 | 3D 检测、速度；可配合跟踪 |
| Cylinder3D | LiDAR | 柱坐标稀疏 3D 网格 | 点云语义分割 |
| BEVFormer | 多相机与历史状态 | 稠密 BEV query/特征 | 3D 检测、地图分割 |
| BEVFusion | 多相机＋LiDAR | 两条 BEV 特征分支融合 | 3D 检测、BEV 分割 |
| MapTR | 多相机等支持的模态 | 地图实例 query＋点 query | 矢量地图预测 |
| Sparse4D v3 | 所选配置为六相机 | 稀疏目标 query/anchor 与历史记忆 | 3D 检测与跟踪 |
| SurroundOcc | 多相机 | 稠密 3D volume | 语义占用 |
| GaussianWorld | 多相机与历史 Gaussian | 稀疏语义 Gaussian→占用网格 | 流式语义占用 |

BEV 是表示方式，不是单一任务。检测框、跟踪 ID、地图折线、点云标签和占用体素是不同输出；它们的指标不可横向当作同一种精度比较。下面的优化分析从各自计算结构推导，收益均为待测假设。

**五、BEVFusion 的论文思路与实现**

MIT 版 BEVFusion 要解决的是不同传感器的表示不一致。相机特征丰富但缺深度，LiDAR 几何明确但稀疏。把相机特征只挂到 LiDAR 点上会丢掉大量图像信息，因此分别将两条分支变为 BEV，再融合。[7]

相机分支按 LSS 思路预测每个特征像素的离散深度概率。对于相机 n、像素 u、深度 d 和通道 c，贡献为 \(P_{n,u,d}F_{n,u,c}\)。通过内外参将该像素/深度位置映射到 BEV cell，累加同一 cell 的贡献：

\[
B_{j,c}=\sum_{(n,u,d):\pi(n,u,d)=j}P_{n,u,d}F_{n,u,c}.
\]

实际实现还处理高度分桶及折叠。LiDAR 分支用体素特征和稀疏编码产生 BEV，随后 concat 与卷积融合，再由任务 head 输出框或语义图。选定部署包中的检测 head/config 要以实际实现为准，不能假定所有 BEVFusion head 相同。

最重要的计算问题是相机视锥特征体积。以六相机、32×88 特征、118 个深度 bin、80 通道为教学例子，显式张量有 159,498,240 个元素，单份 FP16 约 304.22 MiB、FP32 约 608.44 MiB。这是形状计算，不是模型峰值显存实测。

原论文的高效 pooling 使用预计算与区间归约；BEVPoolv2 进一步避免显式构造/预处理大视锥特征，在 kernel 内根据索引读取 depth 与 feature 并计算。[7,10] 优化这里减少了算法实现产生的流量，而不是减少相机或 depth bin。

源码阅读入口：
- mit-han-lab/bevfusion：mmdet3d/models/vtransforms/base.py、lss.py、fusion_models/bevfusion.py、ops/bev_pool/src/bev_pool_cuda.cu。
- NVIDIA CUDA-BEVFusion：src/bevfusion/camera-bevpool.cu。

核对的 NVIDIA kernel 名为 bevpool_half_pack10_kernel，已经按 interval 遍历、每线程累加 10 通道、FP16 输入/输出、FP32 累加，并直接读取 depth 与 camera feature。[11] 因此再次“去掉视锥大张量”不能自动当作相对该实现的新贡献。剩余候选包括移除内循环的索引解码/整数除法、减少重复 depth/索引读取、调整通道 tile、区间长度分组、数据布局和 block 大小。

NVIDIA 2026 年 BEVPoolV3 技术说明也强调按 L2 是否容纳工作集选择优化重点。[12] 所有索引、feature、depth 和输出的缓存行为都要在 4090 上实际测量；其他 GPU 的缓存驻留或加速倍数不能直接迁移。

**六、Sparse4D v3 的论文思路与实现**

Sparse4D 直接维护少量目标实例。每个实例包含特征向量与显式 3D anchor，围绕 anchor 生成采样点，投影到多相机、多尺度特征中，做双线性采样与加权聚合，再迭代修正框。高分实例保留到下一帧，按自车姿态和时间/速度对齐，形成递归时序感知。[8]

可用简化式描述图像特征聚合：

\[
f_{q,c}=\sum_{v,l,k}w_{q,v,l,k,g(c)}
\,\mathrm{Bilinear}(F_{v,l,c},\pi_v(p_{q,k})).
\]

v 是相机、l 是特征层、k 是采样点，g(c) 表示通道对应的分组。它是带几何约束的稀疏 gather/reduction，不是标准全局 QKᵀ attention，因此不能直接套 FlashAttention 替换。

v3 新增时序实例去噪、框质量估计和解耦 attention。去噪在训练中利用带噪 GT anchor 稳定学习，不进入正常推理；质量监督区分分类置信度与定位/朝向质量；解耦设计改善目标特征与 anchor 编码在 attention 中的交互。实例身份在推理中随记忆传递，减少独立关联器的依赖。[8]

官方 R50 配置有 900 anchor、600 历史实例、256 维特征，采样点由 7 个固定点与 6 个可学习点组成。[13] 900 个目标 query 不能直接与 40,000 个 BEV query 作速度比例：backbone、采样次数和 decoder 仍有成本。

源码阅读入口：models/blocks.py、instance_bank.py、sparse4d_head.py，以及 ops/src/deformable_aggregation_cuda.cu。[13] 核对的自定义前向使用 float 数据指针，在双线性采样后用 atomicAdd 向实例/通道输出累加。源码已经融合采样和加权，不能把同一融合再次描述为新优化。

可验证的候选是：按实例/通道组织线程，warp 或 block 内先汇总各相机/层/点的贡献后少量写出；复用坐标、权重和相邻通道特征；降低中间重排成本。是否获益取决于 atomic 竞争、访存、并行度和寄存器压力。更改归约顺序需验证数值和跨帧 ID；FP16 支持要覆盖 kernel 接口与类型，不能只打开 autocast。

**七、GaussianWorld 的论文思路与实现**

GaussianWorld 要预测空间哪里有物体及其语义，输出完整占用，而不仅是有限类别目标框。它继承 GaussianFormer 的语义 Gaussian 表示，并将历史场景递归更新：静态部分按自车运动对齐，动态部分根据当前视觉证据调整，新进入视野的区域补入 Gaussian。[9]

每个 Gaussian 描述位置、尺度/协方差、旋转与语义等属性。统一 world layer 通过 Gaussian 间编码、图像 deformable attention 和属性 refinement 更新场景。演化阶段利用动态语义权重控制历史 Gaussian 的位置变化；后续 refinement 仍可调整历史 Gaussian 的其他属性，因此不是“静态属性永远冻结”。

占用 head 对局部 Gaussian 的语义贡献作加权聚合，可用简化式理解：

\[
S_k(x)=\sum_{i\in\mathcal N(x)}
a_i\exp\!\left[-\tfrac12(x-\mu_i)^\top
\Sigma_i^{-1}(x-\mu_i)\right]s_{i,k}.
\]

该式表示语义贡献，最终 empty、激活及类别判定以实际 head/config 为准；它不是已经归一化的概率公式。任务使用 3D Gaussian 表示，但不能据此把它当作生成 RGB 的普通 3DGS 渲染器。

核对的官方流式配置是六相机 864×1600、25,600 Gaussian、200×200×16 输出、amp=False。论文描述额外 temporal feature，而该配置 temporal_feat_dim=0：继承历史 Gaussian 几何/语义状态仍能实现流式推理，但这条额外特征在所选配置未启用。[14] 复现时必须逐项核对论文、config 与执行路径。

源码入口：segmentor/gaussian_segmentor_stream.py、decoder/gaussian_decoder/gaussian_decoder_stream.py、encoder/gaussian_encoder/、head/gaussian_occ_head.py 与 head/localagg/src/。

核对的 local aggregation 流程为：计算每个 Gaussian 覆盖的网格→前缀和→读取总覆盖数回 CPU→生成 Gaussian/voxel 索引对→CUB radix sort→建立每体素区间→每输出点累加语义贡献。[15] 原版并非简单 G×全部 voxel 全连接计算，优化必须对照其已有局部索引。

候选包括复用有安全容量边界的 workspace、减少小标量 D2H 同步、优化索引/排序流量、改善每体素 Gaussian 列表的负载不均与读取复用。Gaussian 位置和尺度会随帧变化，索引通常不能像固定标定的 BEV 映射一样跨帧直接缓存。减少 Gaussian 数量、截断贡献或放大 voxel 都改变配置/语义，须另做精度实验。

显存优化先检查片段驻留和逐帧搬运，保持模型状态。是否能在 24GB 完整运行及热点排序，仍取决于真实资源测量。

**八、经典模型的思路与实现**

**PointPillars，CVPR 2019。** 将点云按 xy 划成竖直 pillar，柱内 PointNet 提取特征，经 max 聚合成一个向量，再 scatter 到 BEV 伪图像，使用成熟 2D CNN 和 anchor 检测 head。原始点特征含 xyz/强度、柱内均值偏移和柱中心偏移；保留 z 作为特征，而不是简单删掉高度。[16] 原始 KITTI 协议与本项目 nuScenes 适配应分别理解。实现看 PillarFeatureNet、PointPillarsScatter。优化候选是体素分配、统计/特征构造融合、柱内归约、scatter 和后处理。已聚合后的 pillar 通常对应唯一 cell，不应不分场景地加入 atomic。改变每柱点数上限或采样规则会改变实际模型输入。

**CenterPoint，CVPR 2021。** 在 BEV 上检测中心热图，再回归中心偏移、z、尺寸、sin/cos yaw 和速度。它主要改变检测表示和 head，backbone 可用 pillar 或 voxel 编码，因此与 PointPillars 不属于完全互斥的同一层级选择。[17] 原论文还讨论二阶段 refinement，常用配置未必启用。跟踪在统一参考坐标系中用当前中心减去速度×时间间隔回推，再作类别/距离门限关联。实现看 CenterHead 的 heatmap、回归与 bbox decoder，以及独立 tracker。GPU 候选是体素/稀疏卷积、top-k/gather、解码与 NMS；数十/数百目标的关联是否搬 GPU，要计入传输和启动成本。当前项目 NumPy 跟踪优化仍是 CPU 成果。

**Cylinder3D，CVPR 2021。** 将 xyz 转为 radius/azimuth/z，柱坐标网格的远处 cell 物理尺寸更大，用于适应 LiDAR 随距离变稀的采样；结合非对称 3D 稀疏卷积和点级修正进行语义分割。[18] 输出是原始点标签。实现热点可能是点→体素聚合、稀疏邻接索引、gather/GEMM/scatter 和 voxel→point 映射。不能仅凭 3D 卷积 FLOPs 推断耗时，升级稀疏库也要核对权重布局和输出。

**BEVFormer，ECCV 2022。** 为 BEV 网格设置 query，沿高度建立参考点并投影到可见相机，由 deformable spatial cross-attention 取得图像特征；通过 temporal attention 访问历史 BEV，生成当前 BEV。[19] 与 LSS pooling 相比，它从 BEV query 查图像特征，后者把图像/深度贡献汇入 BEV；与 Sparse4D 相比，历史状态是覆盖空间的 BEV，而非目标实例。实现重点看 spatial_cross_attention.py、temporal_self_attention.py。优化要关注采样、跨相机重排、历史状态与矩阵计算，不能把整个模块当一个通用 attention kernel。

**MapTR，ICLR 2023。** 直接预测地图元素的折线/多边形。地图实例 query 与实例内点 query 形成层级表示；同一条线的正反顺序、闭合形状的不同起点可以代表相同几何，训练匹配考虑这些等价排列，再监督类别、点位置与边方向。[20] 实例层用二分匹配，点层在合法等价排列中选取，不是任意打乱点就等价。部署输出为矢量点集。Hungarian/点序匹配主要是训练成本，不是本轮推理热点；应优化 BEV encoder、deformable 采样与 decoder。减少 query/点数是模型配置改变。

**SurroundOcc，ICCV 2023。** 从多相机特征通过 2D→3D attention 得到 volume，用 3D 卷积逐级恢复占用。论文同时给出用多帧 LiDAR、动态/静态分离及重建生成稠密占用标签的方法。[22] 与 GaussianWorld 的区别是场景内部表示及时间更新方式。LiDAR 生成训练标签不等于推理时使用 LiDAR；未观测空间也不等于已证实 empty。优化候选是 3D feature 的流量、采样、卷积和生命周期。

**YOLO 与 YOLOv8。** 项目的 YOLOv8s 属于单阶段 2D 检测实现，利用多尺度卷积特征、检测 head、解码和通常的 NMS 输出框。官方明确没有单独发布 YOLOv8 正式论文，学习以对应版本源码/配置为准，不能为它虚构论文结构。[23] 优先使用 TensorRT 的标准卷积与融合能力；另测预处理和 NMS。手写成熟卷积通常不是第一项工作。

**DETR，ECCV 2020。** 将检测写成固定数量 object query 的集合预测，训练以 Hungarian 匹配分配 GT，结合分类和框回归损失，原始推理无需传统 NMS。[24] Deformable DETR 改为少量参考点采样，是后续几类几何 query 模型的重要基础。训练分配、跟踪关联和官方 AP 匹配是不同用途，不能互换。标准 attention 可以检查成熟优化路径，deformable sampling 另行处理。

**SegFormer，NeurIPS 2021。** MiT 编码器输出多尺度特征，通过高效 attention 和 Mix-FFN 的局部卷积表达空间信息，轻量 MLP decoder 融合得到像素类别。[25] 推理时要把滑窗、图像恢复、attention/矩阵计算、上采样分别计时。FlashAttention 是否可用取决于实际 attention 形式和接口，不能假定对任何带 Transformer 名称的模块都适用。

**Depth Anything V2，NeurIPS 2024。** 利用高质量合成深度训练教师，再以大量真实图像的伪标签训练学生，配合视觉 backbone/深度 decoder 进行单目深度预测。[26] 通用相对深度与米制深度权重不同；本项目 Metric Depth 的误差还与目标域和尺度有关。部署优化可以关注 backbone、decoder、输入尺寸和输出恢复，不能通过事后尺度拟合掩盖参考模型误差。

**九、当前项目应怎样选择第一个 GPU 算子**

现有离线测量中，导入/元数据/权重加载占混合工作量的 53.2%，绘图/视频占 13.9%；这些不是统一稳态流水线的占比。CenterPoint 常驻预热探针约 25.22 ms/帧，六相机深度/SegFormer 的排除首次调用诊断约 511/325 ms 每时刻，但样本与计时边界不同。[21] 因此先改善常驻和导出边界，再获得新部署模型的逐 kernel 证据。

| 候选 | 已知代码特征 | 先测什么 | 首个可验证改动 |
|---|---|---|---|
| BEV pooling | 有 interval、通道 tile、索引解码；NVIDIA 版已融合 depth×feature | 时延占比、interval 分布、L2/DRAM/指令、寄存器 | 预计算明确 feature index，或调整 block/tile |
| Sparse4D deformable aggregation | 前向 float 采样与 atomicAdd 累加 | atomic 竞争、gather 效率、有效视图/采样分布 | 按输出组织局部 reduction，或优化通道访问 |
| GaussianWorld local aggregation | 动态覆盖列表、前缀和、排序、D2H 计数 | 排序/同步/分配/聚合分别占比 | 安全复用 workspace、减少同步 |
| PointPillars 前处理 | 分桶、均值、偏移、归约与 scatter | 各阶段的启动/流量与点数分布 | 融合统计/特征构造，保留选点语义 |

若目的是学习并形成可展示的算子开发成果，BEV pooling 很适合：输入输出和 reduction 语义明确，容易保存真实测试输入。但是否优先于其他项，应由绝对耗时与上界决定。

最小验证流程是：捕获真实输入→参考与候选正确性→空/边界/长短区间/真实最大 shape→独立预热→至少 30 轮交替配对计时→整模型同协议评估→持续场景回放。涉及反向时另外检查梯度，推理优化不必为未修改的反向设计训练实验。

证明成果时同时报告改动前后实现、数据/shape/dtype、GPU/版本、kernel 毫秒数、整分支毫秒数、显存、数值误差与任务指标。若只编译或调用上游 kernel，写“部署上游算子”；只有实际修改、正确性与实测收益支持时，才写“开发/优化 CUDA 算子”。

**十、论文与源码入口**

[1] [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)。

[2] [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) 与 [Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/)。

[3] [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

[4] [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

[5] [TensorRT Performance Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)。

[6] [TensorRT Advanced Topics](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html)。

[7] [BEVFusion 论文](https://arxiv.org/abs/2205.13542) / [MIT 作者页](https://hanlab.mit.edu/projects/bevfusion) / [源码](https://github.com/mit-han-lab/bevfusion)。

[8] [Sparse4D v3 论文](https://arxiv.org/abs/2311.11722)。

[9] [GaussianWorld CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.html) / [论文正文](https://arxiv.org/html/2412.10373v1)。

[10] [BEVPoolv2](https://arxiv.org/abs/2211.17111)。

[11] [NVIDIA camera-bevpool.cu](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/src/bevfusion/camera-bevpool.cu)。

[12] [NVIDIA BEVPoolV3 技术说明 2026-06-24](https://developer.nvidia.com/blog/accelerating-bev-pooling-on-nvidia-gpus-for-physical-ai-applications/)。

[13] [Sparse4D 源码](https://github.com/HorizonRobotics/Sparse4D) / [R50 配置](https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py) / [CUDA 聚合](https://github.com/HorizonRobotics/Sparse4D/blob/main/projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu)。

[14] [GaussianWorld 流式配置](https://github.com/zuosc19/GaussianWorld/blob/main/config/nusc_surroundocc_stream_eval.py)。

[15] [GaussianWorld local aggregation](https://github.com/zuosc19/GaussianWorld/blob/main/model/head/localagg/src/aggregator_impl.cu) / [前向 kernel](https://github.com/zuosc19/GaussianWorld/blob/main/model/head/localagg/src/forward.cu)。

[16] [PointPillars 论文](https://arxiv.org/abs/1812.05784) / [MMDetection3D PillarFeatureNet](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/voxel_encoders/pillar_encoder.py) / [scatter](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/middle_encoders/pillar_scatter.py)。

[17] [CenterPoint 论文](https://arxiv.org/abs/2006.11275) / [作者源码](https://github.com/tianweiy/CenterPoint) / [MMDetection3D CenterHead](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/centerpoint_head.py)。

[18] [Cylinder3D 论文](https://arxiv.org/abs/2011.10033) / [作者源码](https://github.com/xinge008/Cylinder3D)。

[19] [BEVFormer ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/694_ECCV_2022_paper.php) / [作者源码](https://github.com/fundamentalvision/BEVFormer)。

[20] [MapTR 论文](https://arxiv.org/abs/2208.14437) / [作者源码](https://github.com/hustvl/MapTR)。

[21] [CARperception pipeline_timing](../pipeline_timing/report.md) / [部署指令](../../instruction/20261008_01_4090_perception_deployment.md)。

[22] [SurroundOcc ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wei_SurroundOcc_Multi-camera_3D_Occupancy_Prediction_for_Autonomous_Driving_ICCV_2023_paper.html) / [作者源码](https://github.com/weiyithu/SurroundOcc)。

[23] [YOLOv8 官方说明](https://docs.ultralytics.com/models/yolov8/)。

[24] [DETR 论文](https://arxiv.org/abs/2005.12872) / [作者源码](https://github.com/facebookresearch/detr)。

[25] [SegFormer 论文](https://arxiv.org/abs/2105.15203) / [作者源码](https://github.com/NVlabs/SegFormer)。

[26] [Depth Anything V2 作者页](https://depth-anything-v2.github.io/) / [作者源码](https://github.com/DepthAnything/Depth-Anything-V2)。

**源码核对版本**

| 仓库 | 核对的 main/master commit | 关键文件 blob SHA |
|---|---|---|
| HorizonRobotics/Sparse4D | 249ffbb695f4e9db628d953e2bf6d36de04bbb69 | deformable_aggregation_cuda.cu：4f748e55ea0c8374f4935936d63eaca3cc3d1c1c；R50 config：fdf075bf9e739200218d1f70662961d0cddf85dd |
| zuosc19/GaussianWorld | b43629eaecffd5a7cbaac1a55517766e6263e4fc | aggregator_impl.cu：c563d60bf845e8b08ac697e313bd882f6b571a6f；stream_eval：166bdf45ab1ae5a6b4fe62d675e98baa56f02992 |
| NVIDIA-AI-IOT/Lidar_AI_Solution | 7c1623f234d5a9794fd862c9456f68f87ae7b900 | camera-bevpool.cu：55122e10bccf544927cb0f64d8e80357049020ea |

当前源码事实、论文设计和候选优化分别注明。主分支滚动更新后，以此版本或对应 blob 对照；这些源码读取不能替代本机 kernel profile。
