# 实车自动驾驶技术与 CARperception 模型适用性调研

调研日期：2026-10-08（北京时间）。用途：判断当前项目的产业相关性、选择后续学习与部署重点、校准简历表述。

本报告优先采用车企/自动驾驶公司官网、投资者关系公告、官方工程文档及论文作者代码。公开资料核对不等于车端实测；所有升级路线均为建议，本次没有新增模型部署或实验结果。

## 1. 结论

**现有项目以经典模块为主，可以展示感知工程基础，但不足以代表 2026 年领先高阶辅助驾驶的完整架构。** PointPillars、CenterPoint、Cylinder3D、SegFormer 的主要论文发表于 2019—2021 年，MapTR、YOLOv8 属于 2023 年左右的方案，Depth Anything V2 属于 2024 年。[1–6]

**CenterPoint 并没有失去工程价值。** 当前 Autoware 官方文档仍提供 PointPillars 编码器结合 CenterPoint 检测头的 TensorRT 节点，以及相机/LiDAR BEVFusion 的 TensorRT 节点。这能证明这些架构仍用于公开自动驾驶工程栈，不能据此断言某家乘用车品牌使用了本项目的原始权重。[7–8]

**领先车企公开的升级重点已经扩展到时间、联合任务和驾驶决策。** 2026 年资料中可核对 Tesla 的端到端基础模型、Xpeng VLA 2.0 的实际推送、Li Auto 的 MindVLA 与 3D ViT Encoder 上车、Huawei WEWA 2.0，以及 NIO 的 NWM；Waymo 则公开了时序多传感器融合、驾驶 VLM、世界解码器及独立轨迹验证的组合。[9–17]

对本项目的判断是：保留经典基线和已有诊断工具，优先补齐真实多模态融合、时序感知、占用预测与推理部署；按照求职岗位再扩展端到端规划。仅更新二维检测器版本，无法补齐这些能力。

## 2. 哪些技术已有实车证据

证据口径：**A** 为官方明确披露已交付/推送/运营；**B** 为官方产品或在用架构介绍，能够说明公开技术路线，但没有在本次核对中锁定所有车型的实际软件版本；**C** 为发布或商用计划；**D** 为开源工程实现或研究，不能等同于量产采用。分级描述的是本报告拿到的证据，不是算法质量或自动驾驶等级。

| 企业/工程栈 | 核对资料与时间 | 可以确认的技术 | 证据及边界 |
|---|---|---|---|
| Tesla | 2026-01-28、2026-04-22 股东材料；当前 FSD 支持页 | FSD v14 的端到端基础模型；4 月 v14.3 升级视觉编码器、强化学习阶段及编译/运行时 | A；材料说明已有发布版本。这里引用 v14.3 作为可核对实例，不宣称它是 10 月的最新子版本。FSD (Supervised) 仍需驾驶员监督。[9–11] |
| 小鹏 | 2026-10-01 官方交付公告；2026-09-15 技术介绍 | 公告确认 9 月 22 日开始在中国推送 XOS 6.3.0 / VLA 2.0；技术介绍涉及 30 秒历史上下文、流式推理、缓存及未来场景预测 | A；实际推送已确认，但不等于所有硬件、车型或地区均已得到同一版本。[12–13] |
| 理想 | 2026 年第一季度业绩公告中的 5 月产品进展 | 新 L9 已开始交付，搭载 MindVLA、3D ViT Encoder；官方披露自研 M100 芯片与模型集成上车 | A；可确认产品和组件名称，不能恢复私有模型内部全部实现。[14] |
| 蔚来 | 当前 ES8 五座版官方产品页 | NWM 被描述为多变量自回归生成模型，配合闭环强化学习；AQUILA 包含激光雷达与 4D 成像雷达 | B；产品页能确认配置/路线，具体可用功能仍依版本和交付条件，未把产品页当作全部车辆完成 OTA 的证据。[15] |
| 华为乾崑 | 2026-04-23 发布会；2026-07-16 官方媒体日 | ADS 5 / WEWA 2.0：云端世界引擎、多智能体博弈与强化学习；车端世界行为模型和安全风险场 | C；7 月官方表述为即将陆续 OTA 商用。本报告未用第三方转载来补成所有车型实际推送的结论。[16–17] |
| Waymo | 2026-02 第六代 Driver 无人运营公告；2025-12、2026-08 架构文章 | 在用系统包含相机/LiDAR/radar 的时序融合；快反应感知与慢语义推理；教师模型蒸馏为车端学生模型；独立轨迹验证与闭环仿真 | A/B；有无人运营及官方架构证据。2024 年 EMMA 属于研究，不应直接叫作当前量产 Driver。[18–21] |
| Autoware | 当前官方 CenterPoint、BEVFusion 文档 | CenterPoint + PointPillars 编码；相机/LiDAR BEVFusion；ONNX→TensorRT；ROS 2 消息与延迟调试输出 | D；是可学习、可部署的公开工程实现，不能推导出全部车企的量产选型。[7–8] |

上述资料支持“技术方向已变”的结论，不支持编造车企内部 backbone、层数、权重、量化精度或源码。厂商使用“端到端”“VLA”“世界模型”等名称，也不意味着它们实现了同一种网络。

## 3. 现在的实车数据与任务

**可见光相机、激光雷达和毫米波雷达仍是有效的实车传感器组合，组合依企业和车型而异。** Waymo 第六代的运营资料明确说明三者协同；NIO 当前 AQUILA 配置包含激光雷达和 4D 成像雷达。Tesla 则有以相机为核心的 Tesla Vision 路线；其部分地区 Model 3/Y 支持页明确说明无 radar，不能把这一表述外推到所有年代、地区和车型。[15,18,22]

需要区分 LiDAR（激光雷达，几何点云）与 radar（通常指毫米波雷达，距离/径向速度等测量）。VLA 是视觉—语言—动作的一类模型范式；厂商同名产品未必在车端逐字生成语言再控车。BEV 是鸟瞰表示，既可以是稠密网格，也可以配合稀疏实例表示；不是某个单独网络的名称。

| 层次 | 输入/输出与作用 | 本项目对应情况 |
|---|---|---|
| 同步与几何 | 图像、点云、雷达测量、时间戳、内外参、自车位姿/状态；对齐时刻和坐标系 | 已做标定与位姿变换；雷达尚未进入现有模型闭环，动态目标曝光时刻外推未完成 |
| 三维场景感知 | 检测框、类别、速度、可行驶空间、道路元素、语义/特征 | CenterPoint、Cylinder3D、MapTR 已有输出；分别运行，不构成相机—LiDAR 特征融合 |
| 时序理解 | 跨帧目标状态、遮挡恢复、历史特征/实例记忆 | 已有外置 PubTracker 和历史 sweeps；尚无统一的学习式时序感知分支 |
| Occupancy / 占用 | 描述三维空间的占用/空闲与语义，覆盖部分难以由类别框表达的形状 | 尚未实现。逐点语义分割只标记有观测点的位置，不等同于稠密占用预测 |
| 世界模型与预测 | 建模场景如何随时间、动作和交通参与者变化；可服务于预测、训练和仿真 | 尚未实现。一次深度估计、静态三维重建或 MapTR 地图输出均不等同于这一能力 |
| 规划与验证 | 生成或评分自车候选轨迹，结合规则/风险与车辆约束检查 | 尚未实现，当前回放没有车辆行为反馈闭环 |

表中前三列任务的区分结合 Waymo 已公开架构及本项目实现作出，不代表每个量产系统都具备独立同名模块。[19–20] “端到端”可以描述联合训练和信息传递，车端仍可能输出对象/地图、使用安全验证，并存在多模型协作。

## 4. 现有模型是否值得继续保留

这里“经典”“应补充”是对岗位覆盖与项目结构的判断，不是未运行对比实验就宣称某个模型精度更差。

| 当前模型 | 年代/来源 | 判断 | 建议定位 |
|---|---|---|---|
| PointPillars | CVPR 2019 [1] | 较早；柱状点云编码仍有部署价值，Autoware 的 CenterPoint 实现仍使用此类编码 | 保留为轻量 LiDAR 基线，学习编码、scatter、部署与延迟 |
| CenterPoint | 2020 预印本 / CVPR 2021 [2] | 经典，但有当前 Autoware TensorRT 工程证据 | 保留主基线；补正式跨场景评估和部署验证，而非只因年份删除 |
| PubTracker / CPU 候选跟踪器 | CenterPoint 体系中的简洁外置关联 [2, P1] | 便于解释关联与计算优化；不足以覆盖新的学习式时序方案 | 保留固定输入对照，同时增加时序检测/跟踪参考 |
| Cylinder3D | CVPR 2021 [3] | 可展示点云分割与稀疏卷积适配，当前旧依赖维护成本已有实例 | 保留已有成果；优先补 Occupancy，不急于只换另一种逐点分割模型 |
| SegFormer | NeurIPS 2021 [4] | 独立二维分割基线仍能用于教学；难以单独代表现代三维场景理解 | 保留辅助分支，减少重复扩充独立二维模型 |
| YOLOv8s | 官方 2023 年发布 [5] | 不是当前 Ultralytics 新版本；2026 年已有 YOLO26，但改版本只覆盖二维检测 | 可增加 YOLO26 的同口径导出/精度/时延对照，优先级低于融合及时序 [23] |
| MapTR | ICLR 2023 [6] | 矢量地图任务仍相关；当前项目缺少该任务的正式精度和时序验证 | 先评估现有输出；再选择 MapTRv2 或联合地图/跟踪方案，避免只增加展示数量 |
| Depth Anything V2 Metric Depth | NeurIPS 2024 [24] | 比部分现有基线新，但通用单图米制深度不等同于融合感知所需的深度/空间建模 | 保留诊断；检查相机域、尺度与 LiDAR 参考，新增 BEV 分支时依其官方深度机制 |
| Python/NumPy CPU 关联优化 | 本项目 2026 年实现 [P1] | 类别门控、距离矩阵和内存复用具有工程价值；仍属于 CPU 实现 | 保留 3.77× 的局部实测；若面向 GPU 算子岗位，再补 CUDA/TensorRT 算子实测 |

本项目目前的量化成绩来自 nuScenes mini 单场景；多模型在同一界面展示不等于融合；ONNX 输出核验不等于已完成 TensorRT 或车端部署。这些证据边界比网络名称更影响简历可信度。[P1–P2]

## 5. 与当前方向接轨的公开方案

选择依据是与当前项目的接口、任务和复现资源，而非只按发表年份排序。以下作者文档、配置/权重入口已核对，本次没有验证环境能安装、权重能完整下载或本地可达到作者性能。

| 方向 | 建议参考 | 已核对的资源与技术 | 对当前项目的意义 / 边界 |
|---|---|---|---|
| Camera–LiDAR 融合部署 | Autoware BEVFusion；NVIDIA CUDA-BEVFusion [8,25] | 相机/LiDAR 融合节点、TensorRT/稀疏卷积后端；CUDA 部署实现 | 最直接补 JD 中的融合感知和推理优化；虽不是新论文，却有现行部署接口 |
| 多相机时序检测与跟踪 | Sparse4D v3；Sparse4D 2026 年整合论文/仓库 [26] | 跨帧实例传播、检测与跟踪；官方配置/权重；多视图特征聚合 CUDA 实现 | 补学习式时序能力；v3 本身是 2023 年预印本，不将 2026 年整合论文包装为 v3 新发明 |
| 流式三维占用 | GaussianWorld，CVPR 2025 [27–28] | 作者提供流式/单帧配置、权重和评估入口；使用 nuScenes 与 SurroundOcc 占用标注 | 补三维空间语义及时序占用；不能用当前 LiDARSeg 标签直接替代它的占用 GT |
| 感知到规划的一体化理解 | SparseDrive，ICRA 2025 [29] | nuScenes 多相机、实例记忆，联合检测/跟踪/地图及预测规划 | 适合从当前感知实验向规划延伸；公开研究成绩不是量产采用证据 |
| 较新的端到端规划 | SparseDriveV2，2026，作者标注 ECCV 2026 接收 [30–31] | 轨迹词表分解与评分；NAVSIM 配置/权重；另有 Bench2Drive 分支 | 更适合规划岗位；V2 不只是给当前 mini 感知链换网络，NAVSIM/Bench2Drive 需独立数据和协议 |
| VLA 与教师模型路线 | NVIDIA Alpamayo 系列，2026 [32] | 官方 2026-05-31 介绍 34B Alpamayo 2 Super、闭环训练/仿真、自动标注与蒸馏 | 学习大模型如何服务数据和车端学生模型；官方材料将其定位为教师模型，不能据名称认定完整 34B 已在乘用车实时运行 |

**面向截图中的感知模型实习，优先选择前面三项；端到端规划和 VLA 作为后续扩展。** 感知实习要求中的标定、Bad case、三维检测/分割、BEV 与融合，以及部署优化，都能通过这一顺序形成较具体的项目成果。具体车企的私有结构仍需入职后了解，不能靠复现一篇论文完全替代。

## 6. 算子开发如何升级

现有代码优化的是 CPU 跟踪关联，已经有固定检测输入、输出精确一致及配对计时证据。可继续写入简历，但应保留 CPU 与计时边界。[P1]

后续更贴近感知网络部署的算子选择包括：

| 算子 | 作用 | 建议的公开起点 | 验证重点 |
|---|---|---|---|
| Voxelization / Pillar scatter | 将不定长点云转换为体素/柱状或 BEV 特征 | Autoware CenterPoint；NVIDIA 点云部署实现 [7,25] | 点归属、越界/空体素、输出布局、重复点处理及实际前后处理开销 |
| BEV pooling / scatter-reduce | 把多相机深度加权特征汇聚到鸟瞰空间 | NVIDIA CUDA-BEVFusion、官方 BEV pooling 技术博客 [25,33] | 数值误差、索引/内存流量、不同真实 shape、FP32/FP16 与完整分支时延 |
| Deformable feature aggregation | 按三维实例关键点采样多相机/多尺度特征并聚合 | Sparse4D 官方 CUDA 实现 [26] | 坐标采样与边界语义、参考实现一致性、历史实例更新、并行聚合开销 |

NVIDIA 当前 BEV pooling 技术资料讨论了不同 GPU 缓存容量下的内存瓶颈、CUDA 实现、TensorRT IPluginV3 及 Nsight Compute 验证。[33] **因此算子优化方向仍然相关，但 CPU 跟踪函数提速与 GPU 感知算子提速是不同成果。** 引用实现、重新部署、修改算子和独立开发算子，应分别记录实际完成内容。

不要直接搬用官方博客的加速倍数：硬件、TensorRT/CUDA 版本、输入 shape、精度、测量范围都会影响结果。本项目下一步若选择 BEV pooling，应先让完整融合分支在参考实现上运行，再比较替换算子后的正确性、局部时延和整体时延。

## 7. 建议实施顺序和验收

这是后续工作建议，尚未派发实施指令，也不表示以下验收已完成。

| 顺序 | 工作 | 可检查的交付 | 指标/边界 |
|---|---|---|---|
| P0 | 扩展现有基线评估与推理计时 | 多场景/官方 val 预测、官方评估日志、常驻进程计时与失败案例 | 检测 mAP/NDS；跟踪官方 AMOTA/AMOTP；记录冷启动、预处理、推理、后处理及 P50/P95。mini 仅做联调 |
| P1 | 接入真实 Camera–LiDAR BEVFusion | 六相机、点云、标定真实进入融合网络；同步可视化；环境与权重记录 | 检测 mAP/NDS；内存和分段时延。若声称融合收益，需匹配的数据/训练协议和相应单模态对照，不能拿不同权重直接归因 |
| P2 | 接入 Sparse4D v3 时序检测/跟踪 | 连续场景推理、明确场景切换时重置记忆、目标轨迹与遮挡案例 | 官方检测/跟踪指标。相机模型与 LiDAR 基线可作系统对照，但模态差异不能归因于时序模块；需要自身时序对照 |
| P3 | 接入 GaussianWorld 占用分支 | 作者标注协议、单帧/流式输出、占用体素与语义可视化 | 同一占用协议的 IoU/mIoU、可见性/忽略 mask。SurroundOcc 与其他占用基准不能混报 |
| P4 | 完成一个 GPU 算子与 TensorRT 优化闭环 | 参考输出、优化算子、误差检查、真实 shape benchmark、整分支结果 | 报局部和整体时延、显存、精度。计时显式同步；预热与稳态测量；未测车端不称车端成绩 |
| P5 | 按求职方向再扩展规划/VLA | SparseDrive 或 SparseDriveV2 的独立运行与评估；资源满足后再试 VLA | nuScenes 开环轨迹指标、NAVSIM PDMS/EPDMS 或 Bench2Drive 指标分别按作者协议；开环回放不能冒充交互闭环驾驶 |

若时间只允许一个扩展，选择 **BEVFusion + 推理部署/算子剖析**。若允许两个，再补 **时序检测与跟踪**。这一优先级是针对现有工程和目标感知岗位作出的建议，尚不是实验比较结果。

## 8. 对简历与面试的影响

当前简历可以继续突出多传感器数据适配、CenterPoint 检测与跟踪、点云分割、MapTR、官方评估、Bad case 诊断和 CPU 关联优化。它们能够展示坐标处理、工程排障、评价协议和正确性保持等具体能力。[P1]

**当前定位宜保持“感知部署与关联优化项目”。** 新方案真正跑通并评估后，才能增加“融合感知”“时序三维感知”“占用预测”“CUDA/TensorRT 算子优化”等成绩。单场景 3.77× CPU 提速不代表全系统提速；目前也没有可据以宣称量产车端部署的证据。

面试可以这样回答技术年代问题：“我先用经典模型建立可解释、可复现的感知与评估流程，在固定输入下验证了跟踪策略和 CPU 关联实现的优化。现有公开量产资料已经强调时序、融合和联合决策，因此下一阶段重点是融合及时序感知，并把部署和算子优化补成可测量的完整分支。”最后一句是计划，讲述时须与已完成工作分开。

## 9. 来源与阅读入口

全部网页于 2026-10-08 核对。年份标注表示论文会议/官方公告时间；滚动更新的文档和产品页不假定有固定发布日期。只使用下列原始发布者或作者来源。

### 经典模型与工程证据

- [1] [PointPillars，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html)。
- [2] [Center-Based 3D Object Detection and Tracking，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html)。
- [3] [Cylinder3D，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Cylindrical_and_Asymmetrical_3D_Convolution_Networks_for_LiDAR_Segmentation_CVPR_2021_paper.html)。
- [4] [SegFormer，NeurIPS 2021](https://papers.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)。
- [5] [Ultralytics YOLOv8 官方文档](https://docs.ultralytics.com/models/yolov8/)。
- [6] [MapTR 作者仓库，ICLR 2023 / 后续 MapTRv2](https://github.com/hustvl/MapTR)。
- [7] [Autoware CenterPoint：当前官方节点文档](https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_lidar_centerpoint/)。
- [8] [Autoware BEVFusion：当前官方节点文档](https://autowarefoundation.github.io/autoware_universe/latest/perception/autoware_bevfusion/)。

### 量产/运营与企业架构

- [9] [Tesla 2025 Q4 股东材料，发布于 2026-01-28](https://ir.tesla.com/_flysystem/s3/sec/000162828026003837/tsla-20260128-gen.pdf)，AI & Software。
- [10] [Tesla 2026 Q1 股东材料，发布于 2026-04-22](https://ir.tesla.com/_flysystem/s3/sec/000162828026026551/tsla-20260422-gen.pdf)，AI & Software。
- [11] [Tesla 当前 FSD (Supervised) 支持页](https://www.tesla.com/support/fsd)。
- [12] [小鹏 2026 年 9 月与第三季度交付公告，2026-10-01](https://ir.xiaopeng.com/zh-hant/node/10226/pdf)，明确记录 2026-09-22 开始推送。
- [13] [小鹏 XOS 6.3.0 技术解读，2026-09-15](https://www.xiaopeng.com/news/company_news/5593.html)。
- [14] [理想 2026 年第一季度业绩公告](https://ir.lixiang.com/news-releases/news-release-details/li-auto-inc-announces-unaudited-first-quarter-2026-financial/)，Recent Developments / All-New Li L9。
- [15] [蔚来 ES8 五座版官方产品页](https://www.nio.com/es8-five-seater)，AQUILA 与 NIO WorldModel。
- [16] [华为乾崑技术大会，2026-04-23](https://auto.huawei.com/cn/news/2026/2026-04-23-jishu)。
- [17] [华为乾崑媒体日，2026-07-16](https://auto.huawei.com/cn/news/2026/2026-7-16-media-open-day)。
- [18] [Waymo 第六代 Driver 开始无人运营，2026-02](https://www.waymo.com/blog/2026/02/ro-on-6th-gen-waymo-driver/)。
- [19] [Waymo Foundation Model 架构说明，2025-12](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/)。
- [20] [Waymo 实际无人运营中的 AI 经验，2026-08](https://blog.waymo.com/blog/2026/08/10ailessons/)。
- [21] [Waymo EMMA 研究介绍，2024-10-30](https://waymo.com/blog/2024/10/introducing-emma/)。
- [22] [Tesla Vision 与部分地区车型传感器说明](https://www.tesla.com/en_GB/support/autopilot)，结论仅按原文适用地区/车型解读。

### 更新模型与算子资料

- [23] [YOLO26 官方发布，2026-01-14](https://www.ultralytics.com/news/ultralytics-redefines-state-of-the-art-vision-ai-with-yolo26) / [官方模型文档](https://docs.ultralytics.com/models/yolo26)。
- [24] [Depth Anything V2，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/26cfdcd8fe6fd75cc53e92963a656c58-Abstract-Conference.html)。
- [25] [NVIDIA CUDA-BEVFusion 官方部署实现](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/README.md)。
- [26] [Sparse4D 官方仓库：v1/v2/v3、配置、权重与 CUDA 聚合](https://github.com/HorizonRobotics/Sparse4D)。
- [27] [GaussianWorld，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.html)。
- [28] [GaussianWorld 作者代码与数据/权重说明](https://github.com/zuosc19/GaussianWorld)。
- [29] [SparseDrive 作者仓库，ICRA 2025](https://github.com/swc-17/SparseDrive)。
- [30] [SparseDriveV2，2026-03-31 预印本](https://arxiv.org/abs/2603.29163)。
- [31] [SparseDriveV2 作者代码与权重、ECCV 2026 接收信息](https://github.com/swc-17/SparseDriveV2)。
- [32] [NVIDIA Alpamayo 2 Super 官方介绍，2026-05-31；含 2026-09 名称更新](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis)。本报告据此讨论公开教师模型路线，不把发布计划当作量产车型部署证明。
- [33] [NVIDIA：Accelerating BEV Pooling on NVIDIA GPUs for Physical AI Applications](https://developer.nvidia.com/blog/accelerating-bev-pooling-on-nvidia-gpus-for-physical-ai-applications)，CUDA/TensorRT 与 Nsight Compute 技术资料。

### 本项目已完成事实

- [P1] [简历项目经历与成果依据](../resume_project_experience_20261008.md)。模型实验依据快照为 `cfe4083664fa1de377d6a84156bdcd42d5defffd`；当前简历文件核对到 blob `da93b33273111e635452a59bb72bae20eb261ef7`。
- [P2] [项目 README](../../../README.md) / [任务状态](../../TASK_STATE.md)。本报告建议不替代现有状态文件，不把建议路线记为已完成任务。
