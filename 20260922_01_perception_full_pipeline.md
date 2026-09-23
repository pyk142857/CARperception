# 给 Codex 的执行指令：自动驾驶感知全流程部署与实验

## 0. 你的任务与完成条件

你需要在当前工作站实际实施本指令，交付可复现工程和真实实验结果。先读项目根目录和上级适用的 AGENTS.md、现有 instruction/，再读同包 perception_execution_plan_20260922.md 与 configs/execution_plan.yaml。本文与 YAML 中定义的默认值属于本次任务要求；现有项目有已确认的用户约束时应一并保留。

目标是跑通 M00–M14，覆盖 10 个真实神经网络、1 个真实跟踪组件、数据几何、部署对比、训练改进、联合可视化与报告。不要止步于分析、生成代码、创建环境或下载权重。逐模块执行、定位错误并修复，直到通过相应验收，或有明确的外部阻碍。

里程碑 A=完整连续场景回放；B=A+规定的任务评估与部署测试；C=B+训练对照与最终报告。只有全部必需项有真实证据时才能声明 C 完成。计划中未列为必需的整套 3D TensorRT 化、INT8、闭环仿真和规划控制不作为本轮完成条件。

持续维护 TASK_STATE.md，每次更新列明已完成项、当前实际运行的命令、失败原因、下一步以及证据路径。需要中断或上下文切换时先保存状态；恢复后继续尚未完成的依赖节点，不重复下载或训练已经校验的资产。除用户明确授权或适用指令要求外，不派生额外代理执行任务。

## 1. 必须遵守的实施规则

1. 运行官方模型或可核对的忠实实现，并加载对应的完整有效权重。保留架构、head、类别和预处理语义；官方 tiny/small 变体已经列在计划中。不能因为报错而换成随机网络、空输出、规则框、玩具检测器、少一条融合分支的网络或其他未声明模型。
2. 安装、运行、评估、导出、导出后验证、训练必须分别记状态。禁止用“代码能导入”代表“模型已运行”，用一张效果图代表“评估已通过”，或用 ONNX 文件存在代表“TensorRT 已部署”。
3. 不使用未明确来源的第三方 checkpoint。记录 repo commit、config、权重 URL、本地 SHA256、标签表、依赖版本和变更 patch。下载来源解析失败时记为 blocked_assets，不伪造链接或权重。
4. 采用小范围兼容性修复处理真实依赖问题。保留修复前后差异，并检查关键算子/权重/输出是否一致。不要盲目执行过时 README 中的 PyTorch/CUDA 组合。
5. 重 GPU 任务串行；正常 CPU 数据整理可并行。按实际硬件编译 CUDA 扩展；环境隔离，避免不同 MMCV/MMEngine/SpConv 版本互相覆盖。不要强制升级用户全局 CUDA/驱动或破坏已有工作环境。
6. 保存原生预测和标准化预测。报告不混用数据划分、标签空间和评估协议；未知指标为 null 并附原因，不用 0 伪装成功。
7. 图像上画出的框、地图线、占用体素必须来自本次运行。真值若参与展示，使用明显不同的样式并标注 GT；真值不能作为模型输入或替换预测。
8. 不用下载到的作者预测文件替代当前模型输出，不把论文指标填成实测值。禁止把 GT 目标 ID、未来检测或未来图像传给在线跟踪/时序推理；历史帧按官方配置处理。
9. 任何缺数据、OOM、算子错误、引擎错误都保留原始日志。独立模块继续执行；依赖失败的模块记 blocked，不把失败吞掉后让整条流程返回成功。
10. 工作限制为当前用户授权的工程与机器。公开资产下载、隔离环境创建、可逆兼容修复按任务推进；账号登录/数据许可缺失时输出准确缺失项，继续其他工作。不要自开付费云资源，也不要擅自上传数据、推送远程代码或部署外网服务。

## 2. 首次预检与工程组织

### 2.1 先探测，再确定执行环境

记录操作系统、GPU 名称和显存、驱动、CUDA runtime/toolkit、Python、PyTorch、磁盘可用空间、内存、CPU、编译器及已有环境。nvidia-smi 显示的 CUDA 上限不等于环境内安装的 toolkit。

定位已有数据、权重和仓库，复用可验证资产。读取各官方仓库的当前安装文档与模型配置，提出满足实际 GPU 的组合，再做真实 import、CUDA 张量运算和所需自定义算子 smoke check。将最终有效版本锁定，而不是把未验证的候选依赖写成成功结果。

至少划分：编排/可视化环境、2D 任务环境、MMDetection3D 环境，以及确有依赖冲突的 legacy 模型环境。可以复用兼容环境；不要求每个模型都一个容器，也不能强行把全部旧模型装进一个环境。

记录磁盘预算：原始数据、解压暂存、模型专用 infos、权重、预测、视频及训练 checkpoint。先完成真实单帧测试，再根据测得时间估算完整验证预算。若空间不足，清理本任务可重建的临时文件或采用按模型处理；不要删除用户数据。

### 2.2 在合适的子目录创建工程

默认项目名 perception_lab。如当前仓库已有同类工程，复用并增量实现，不复制出互相冲突的两套系统。保留现有主业务代码和接口。

建议目录如下；除输入数据与下载权重外，其余必须由实际实现生成：

    perception_lab/
      README.md
      TASK_STATE.md
      instruction/
      configs/                 # 执行计划、模型适配配置
      configs/locked/          # 确切版本、路径和已解析参数
      adapters/                # 各原生仓库输出的薄适配层
      tools/                   # 编排、评估、回放、性能、验收入口
      envs/                    # 已验证的环境与依赖说明
      third_party/             # 固定 commit；保留修改 patch
      data/                    # 数据路径引用；不复制已有大数据集
      checkpoints/manifest.json
      manifests/               # 场景/样本/划分清单
      runs/<run_id>/<module>/  # 互不覆盖的运行与原始输出
      outputs/                 # 可视化与共享回放索引
      results/                 # 汇总指标和状态
      reports/                 # 最终报告和静态 HTML
      logs/

为不同仓库分别准备 infos 缓存，缓存键至少包含 dataset version、split、repo commit、converter config hash。MIT BEVFusion 与较新 MMDetection3D 可能有坐标定义差异，不直接共用 pkl。

先实现最小必要入口，不开发账号系统、云平台或与此任务无关的前端框架。

### 2.3 实现并实际验证统一 CLI

以下是要实现的接口规格，并不是已经存在的命令。创建对应程序后，必须真实运行并记录最终可用命令：

    python tools/run_pipeline.py --plan configs/execution_plan.yaml --phase preflight
    python tools/run_pipeline.py --plan configs/execution_plan.yaml --module M00 --stage all --resume
    python tools/run_pipeline.py --plan configs/execution_plan.yaml --module M01 --stage smoke --resume
    python tools/run_pipeline.py --plan configs/execution_plan.yaml --phase baseline --resume
    python tools/run_pipeline.py --plan configs/execution_plan.yaml --phase evaluate --resume
    python tools/run_pipeline.py --plan configs/execution_plan.yaml --phase deploy --resume
    python tools/run_pipeline.py --plan configs/execution_plan.yaml --phase experiment --resume
    python tools/run_pipeline.py --plan configs/execution_plan.yaml --phase report --resume
    python tools/run_pipeline.py --plan configs/execution_plan.yaml --phase all --resume
    python tools/verify_artifacts.py --plan configs/execution_plan.yaml --milestone C

跨环境通过子进程调用各模型的真实脚本，传递结构化配置和文件路径，不在单个 Python 进程内导入不兼容框架。

依赖按阶段和目标细化：M11 的两个检测来源分别执行；M12 对已就绪模型分别导出/测速；M13 只依赖 M05 的有效基线；M14 即使其他项失败也要生成标记 partial 的诊断报告。汇总模块的某个目标受阻，不能阻止其他已就绪目标运行；全部通过仍要求所有必需目标完成。

resume 必须比较代码、配置、权重和输入清单哈希以及产物完整性，不能只凭目录或 passed 字样跳过。重新运行保存新的 run_id；历史结果不覆盖。

## 3. 数据和统一接口

### 3.1 准备的数据资产

必需：nuScenes mini、nuScenes trainval 所需原始文件和元数据、nuScenes 地图扩展、SurroundOcc 匹配的语义占用标签、COCO 2017 val 图像与 annotations、Cityscapes val 图像和对应精细标签。按实际模型依赖补齐 CAN bus 或其他官方辅助元数据；不能把根本不存在的文件路径写成已准备。

优先复用已有数据。公开可直接下载的资产自动准备，需用户账号接受条款的资产记录在 reports/missing_assets.md：用途、官方入口、准确文件名、期望路径、当前状态。不要把测试集当成有标注的验证集。

三个固定作用域：

- smoke：mini 的固定 1 帧和 1 个连续场景，只验证工程链路。
- replay：官方 val 场景名排序后的前 2 个完整场景，在预测前冻结 token 清单；对全部必需模型使用相同样本序列。额外多样性展示可以追加，但不改写主清单。
- benchmark：各任务完整官方验证协议。nuScenes pilot 可以先取前 20 个 val 场景做调试；所有 pilot 指标标记范围，不替代 benchmark。

M13 微调数据：官方 train 场景按名称排序后取前 80 个，种子 20260922，冻结清单。过滤掉不完整场景并记录排除原因；训练和验证场景集合必须不相交。所有训练抽样/增强只使用训练划分。

### 3.2 FramePacket 语义

每帧至少包括：

- sample_token、scene_token、sample timestamp_us、reference timestamp_us、split。
- ego_ref：以该样本 LIDAR_TOP 时刻的 ego 坐标作为统一回放参考，x 前、y 左、z 上，长度为米。
- 每相机的原图路径、尺寸、sample_data_token、timestamp_us、K、T_sensor_to_ego、T_ego_to_global。
- LiDAR 和每个 radar 的路径、字段、timestamp、外参、对应时刻 ego pose；历史 sweep 单独列出时间差和变换。
- 标签只放入 evaluator/visualizer 的独立数据通路。允许模型使用的额外 metadata 要显式声明。

不能把所有传感器假定为同一采样时刻。跨时刻投影需组合源传感器外参、源 ego pose、目标 ego pose 和目标外参。历史 sweep 使用各自 pose 补偿；转换后才允许累积。

### 3.3 PredictionBundle 语义

所有适配器都产生统一头：run_id、module_id、model_id、sample_token、scene_token、timestamp_us、status、native_coordinate_system、label_namespace、source_config_hash、checkpoint_sha256。

任务体按需包含：

- boxes2d：原图像素坐标 xyxy、class_id、score、camera_channel；COCO 标签不冒充 nuScenes 标签。
- boxes3d：ego_ref 中中心 xyz、size_wlh=[width,length,height]、rotation_wxyz、velocity_xy、class_name、score。每个原生框的重心/底面中心、尺寸顺序、yaw 约定必须显式转换。
- tracks3d：上述框加 tracking_id、轨迹来源及 tracking score；ID 在同场景内稳定，场景间不继承状态。
- segmentation：原图大小的整型类别数组、调色板与 ignore_id；同时保留概率/logits 的可选路径。
- depth：原图大小 float32，单位米，另存 valid_mask；色图只用于展示。
- vectors：ego_ref 米制 polyline、类别、score，保留原生点序与类别定义。
- occupancy：原生三维网格文件、axis order、voxel size、origin、range、类名、valid/unknown/empty 定义。统一回放不得通过随意重采样改写评估网格。

保留模型原生输出以运行官方 evaluator。JSON/JSONL 存元信息与目标列表，NPY/NPZ 存稠密数组，PLY 存点/体素可视化。结果应有真值独立文件；禁止在标准化预测中混入 GT。

合法的“本帧没有检测目标”与模型异常必须可区分。只有前者可以出现成功状态和空框列表；后者为 failed 且有错误摘要与日志路径。

## 4. 逐模块执行与调试

每个模块遵循：资产验证 → 官方入口 → 单帧 → 连续场景 → 适配器核对 → 回放图 → pilot 指标 → 正式评估。先完成所有模块的回放基线，再集中进行完整 benchmark；早期已经完成的正式评估可复用有效缓存。

### M00：数据与几何

使用 nuScenes 官方读取工具，不猜测二进制 reshape 字段。原始 LiDAR 第五字段与多帧模型特征中的时间差不是一回事；radar PCD 交给 devkit 解析，并保存使用的过滤器。

保存同一真实帧的六相机 LiDAR 投影、radar 投影、GT 3D 框投影和 BEV。检查正深度、像素边界、位姿顺序和四元数顺序。几何回环检查使用小误差阈值，同时人工可视化检查墙面、道路和车辆是否合理对齐；数值可逆不能单独证明外参正确。

至少验证跨时刻点投影、框中心与尺寸转换、样本 token 一一对应三个边界。只有 M00 通过，依赖它的统一回放才可通过。

### M01：YOLOv8s 2D 检测

从 Ultralytics 官方发布获取 yolov8s 权重，锁定兼容版本。先原生 predict/val，再封装适配器。检查 letterbox 反变换、xyxy、阈值与 NMS。COCO 正式评估遵循官方 val 配置；展示阈值与评估阈值分开记录。

输出 COCO 指标和 nuScenes 六相机图。对 nuScenes 仅做无 2D GT 的展示时，评估字段为 not_applicable_with_reason，不计算伪 AP。

### M02：SegFormer MIT-B0

从 MMSegmentation 模型表解析 Cityscapes 1024 配置与对应 checkpoint；确认不是 ADE20K 权重。核对 trainId/labelId、19 类语义定义、ignore 标签、归一化和 resize。正式指标遵循原配置的 whole/slide 推理方式。

输出 Cityscapes mIoU、分类别 IoU、混淆矩阵；nuScenes 展示原图+半透明分割+图例。不能把道路颜色渲染出来就声称可行驶区域评估通过。

### M03：Metric Depth Anything V2 Small

使用作者 metric_depth 下的 VKITTI Small 权重，目标文件为 depth_anything_v2_metric_vkitti_vits.pth；模型参数按作者对应配置，户外 max_depth=80 m。不要把 relative depth checkpoint 的输出直接标成米。

将单次 LiDAR 扫描按真实位姿投影到图像，使用相机坐标 Z，而不是点到 LiDAR 的欧氏距离作为深度真值。对同像素多个点保留最近的正 Z，固定 0.1<Z<=80 m、图像边界、有限值等有效条件。不得用未来/多帧点云填成动态场景的“稠密真值”。

使用冻结的 GT 有效掩码，在全部有效像素上累计：AbsRel=mean(abs(pred-gt)/gt)，RMSE=sqrt(mean((pred-gt)^2))，δ1=mean(max(pred/gt,gt/pred)<1.25)。无效、非正或非有限预测单独报错，不能通过删除这些像素改善指标；同时报告每图与全体汇总的像素数量。将其命名为“nuScenes LiDAR 稀疏深度诊断”。不能进行利用验证真值的逐图中位数缩放后再声称 metric 模型本身准确。时间不同步和遮挡可能造成投影误差，分析中写明影响。

### M04：PointPillars

采用 MMDetection3D 官方模型表的 nuScenes SECFPN 2x 配置与匹配权重。检查原始点字段到模型输入字段、历史帧数、pillars、point_cloud_range、voxel size 和 box_type_3d。

模型必须真的执行 pillar 特征、BEV 主干和检测头。输出官方 nuScenes JSON、标准化框、BEV/相机投影和 mAP/NDS。输出缺失速度时按模型与官方提交协议处理并记录，不能捏造物体运动。

### M05：CenterPoint

采用 MMDetection3D nuScenes voxel 0.1、SECFPN、circle NMS 变体与对应权重。运行原生体素化、稀疏卷积、中心热图和回归 head，保存实际配置中的体素大小、点云范围、历史 sweep 数和速度字段。

检查 SpConv 与 CUDA，确认 circle NMS 与配置一致。对框进行原生→统一→原生核对，并将模型预测转换成 nuScenes 官方全局坐标用于评估。该输出作为 M11 的一条检测输入，也作为 M13 初始模型。

### M06：BEVFormer-tiny

采用作者 R50 tiny 官方配置及权重。按照官方模型生成多相机输入、lidar2img、相机增强参数、ego motion 和历史 BEV。记录实际历史信息范围。

同一个场景按 timestamp 顺序推理；场景首帧和切换场景时清空 prev_bev 及相关状态。不能用单帧模式运行后标注成时序配置，不能跨场景复用缓存。若启用 FP16，用同一模型权重做数值检查，不混用作者另一训练 run 的 checkpoint 充当精度消融。

### M07：MIT BEVFusion

首选作者仓库，确认选择 ICRA 2023 MIT BEVFusion 的 camera+LiDAR 检测模型。默认配置定位：
configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
默认权重定位：官方 bevfusion-det.pth。

上述是上游入口提示，使用前必须确认固定 commit 中真实存在并与权重匹配。原仓库有旧依赖，先验证实际 GPU 能运行；不得仅照抄旧 CUDA 环境。

必须实际运行图像分支、LiDAR 分支、BEV pooling、fusion 和检测头。优先利用官方单进程/单卡入口；调整分布式启动方式不等于改模型。若使用官方集成版本替代旧框架，实现差异、模型定义、配置与权重转换必须可核对，并在 run manifest 记录；不暗中换另一篇同名 BEVFusion。

分别检查两模态输入非空且进入对应分支。记录模态覆盖与必要的诊断输出；不能关闭图像分支仍称相机+LiDAR 融合已完成。

### M08：CenterFusion

采用作者 DLA / centerfusion_e60。核对图像预处理、radar 投影、velocity 字段、相机视野、官方过滤器和 3D 框解码。必要时对旧版 DCNv2 做可追踪的兼容移植，并验证 CUDA 真实执行；不能删掉 deformable convolution。

遵循作者 nuScenes converter 及类别协议，将结果交给同一版本 devkit。可视化 radar 的位置、RCS、速度箭头与预测框，说明该输入是 nuScenes 毫米波雷达数据，不是 LiDAR，也不是任意雷达原始波形。

### M09：MapTR

采用作者 MapTR-tiny R50 GKT 24ep 配置和对应 checkpoint。准备其所需 nuScenes map 扩展和转换结果。模型读取相机及允许 metadata；HD map 真值只用于监督/评估，不能画上地图真值就冒充模型预测。

输出官方类别的矢量线，遵循官方距离阈值、点重采样和方向匹配的 AP 协议。保留 road divider、pedestrian crossing、boundary 等实际类别定义；不声称覆盖完整车道连接拓扑或车道中心线功能。

### M10：SurroundOcc

使用作者语义占用预测配置、完整权重和配套标签。记录体素范围、分辨率、轴顺序、类别及空/未知/无效区域语义。不能拿其他占用数据集的标签直接套其 evaluator。

模型预测时输入相机及官方允许的元数据；占用真值生成过程可能使用的多帧 LiDAR 不能被送入纯视觉模型充当额外观测。执行官方语义和几何评估，输出原始体素数组、带类别颜色的 PLY、BEV 和高度切片。未知区域不能涂成“空闲安全区域”。

### M11：3D 多目标跟踪

使用 CenterPoint 作者跟踪实现；从其 nuScenes 跟踪入口解析真实命令。分别读取 M05 与 M07 本次生成的 detection JSON，禁止使用作者下载的预测结果代替。

核对官方检测类与 tracking 类的差别，过滤至官方 tracking 定义；不把全部 10 个检测类别直接当成跟踪评分类别。输入框、速度及时间差单位正确；不依赖 GT ID；空检测帧仍执行跟踪器的时间更新。

输出 AMOTA、AMOTP、ID switches 等及连续轨迹视频。场景切换时完全重置状态，视频使用可区分的 ID 颜色并显示 scene/frame/timestamp。两个检测来源使用固定的同一套跟踪器参数，生成独立结果目录。

## 5. M12：导出、正确性与性能

### 5.1 部署范围

M01、M02、M03 必须尝试并完成 ONNX Runtime GPU、TensorRT FP16 的真实推理和回归检查；若在当前设备上无法修复，记录 blocked/failed，B/C 相应未完成。

M04–M10 本轮必须完成原生 GPU 运行与性能记录，并记录 TensorRT 算子支持调查。完整 3D TensorRT 导出作为扩展，默认不作为 B/C 必需项。M11 的 CPU 关联耗时单列，它是算法实现的实际运行方式，不伪装成 GPU kernel 时间。

对每个导出目标记录 input/output 名称、dtype、shape、opset、动态范围、TensorRT/CUDA/driver 版本、GPU、插件及预后处理位置。TensorRT engine 依赖构建环境，不宣称可以在任意 GPU 直接复用。

先做固定输入尺寸，避免把动态 shape 的问题混入首轮部署。YOLO 默认 B1/640x640；SegFormer 以所选原配置滑窗 crop 为固定引擎输入，并保留滑窗拼接；Depth 以官方预处理后的有效尺寸构建有限的 shape profile。遇到长宽比导致多个尺寸时明确 profile 覆盖，不能未经说明裁剪成不同任务。

### 5.2 导出正确性

必须用相同权重、相同输入和相同预后处理比较 PyTorch、ONNX Runtime GPU 和 TensorRT。逐层或关键输出记录绝对/相对误差及 NaN/Inf；记录 ONNX Runtime provider，GPU 路径不能静默 CPU fallback。

正式任务指标要在相同完整评估清单上复测。默认工程回归门槛是：
- YOLO COCO AP 下降不超过 0.5 个百分点；
- SegFormer mIoU 下降不超过 0.5 个百分点；
- Depth AbsRel 相对恶化不超过 2%。

这些是本任务预设的工程容差，不是行业通用标准。原始指标同时保存，不能四舍五入掩盖下降。预期不满足时调查精度、算子和预后处理；不得在看到结果后随意放宽门槛以通过验收。使用混合精度保护某些层时，完整记录 precision policy 与性能。

对于语义分割，导出 head 成功但缺少滑窗/resize/argmax 后处理，不算整个任务部署完成。对于深度，导出后依然要保证米制尺度和原图尺寸还原。

### 5.3 性能计时

统一规则：
- 单模型 GPU batch=1；真实输入。预热至少 50 次，测量至少 200 次，记录样本数和重复方式。
- CUDA kernel 时间用 CUDA event 或正确同步的计时；记录 CPU 端 wall time，不能把异步提交时间当成推理时间。
- 分别记录模型计算时间、预处理+H2D+模型+D2H+后处理的 warm 服务时间，以及含解码/文件读取的离线回放 wall time。明确哪些开销被包含。
- 冷启动、权重加载、首次 CUDA 编译和 engine build 单列；不混入 warm p50/p95。
- 控制输入尺寸/点数/历史帧范围，记录 GPU 峰值 allocated/reserved 与进程实际占用；不得把只统计 PyTorch allocator 的值当成 TensorRT 总显存。
- 2D 单相机 FPS 和六相机每个同步 sample 的吞吐分别标注；3D 多相机模型按场景 sample 计量。
- 模型计时关闭绘图、MP4 编码和交互界面；可视化耗时另报。整套多模块回放要实测，不通过平均或相加各模型 FPS 推导。
- 多场景统计必须保持时序，warmup 结束后对正式测试场景重置状态；模型级内核 benchmark 和完整序列吞吐分开。
- 只有在实际满足所声明输入频率与全部计算开销时才可说“实时”。把 2 Hz 关键帧渲染成 30 FPS 视频不代表模型达到 30 FPS。

输出 performance.csv，字段至少包括 model/run/precision/shape/batch/sample_count/timing_scope/p50_ms/p95_ms/throughput/peak_memory/hardware。

## 6. M13：一次训练和 bad case 改进

所有模型完成推理基线后，针对 M05 完成有限预算的真实微调实验。先在训练数据/训练预测上选出一个可操作问题，并写 experiment_card.md，说明假设、唯一变更、验证指标、训练预算和停止条件。

初始 checkpoint 固定为 M05 已验证权重。训练数据固定 80 个官方 train 场景；两组分别为：
- A：基线微调。
- B：只改变一个预先登记的训练因素，如类别采样；不得同时改网络、分辨率、优化器和增强。

默认每组 1,000 次 optimizer update，micro batch 1、accumulation 4，学习率为所选官方初始学习率的 0.1 倍；若配置以有效 batch 自动缩放，先禁用冲突的重复缩放并保存最终数值。两组保持相同优化器、种子、总 update、归一化策略与精度设置。微批量下按上游实现选择并固定正常可用的 normalization 策略；必须在训练前登记，不能在 A/B 间改变。

先以小 batch 进行真实 forward/backward，检查梯度非零且有限、optimizer step 后参数发生变化、checkpoint 能重新加载并恢复优化器/调度器。确认后继续到完整预算，不能在 smoke train 后就标记 M13 完成。

记录 update 数量、数据曝光次数、loss、LR、梯度、耗时、峰值显存和每次 checkpoint。初始模型、A、B 用同一官方 val 协议重新评估；报告一次预先登记的对比，不反复根据 val 挑选最佳训练变更。

输出分桶误差与至少若干可核实的成功/失败案例，每个案例给 sample_token、类别、距离/点数等有依据的属性。缺少天气、遮挡等可靠标签时记 unknown，不能靠模型推断标签后冒充人工真值。

训练收益没有保证。没有提升时报告“未观察到提升”及证据，仍可完成这一实验步骤。不能为了好看删掉性能下降或失败结果，也不能把它写成从零复现原论文精度。

## 7. M14：可视化与最终报告

### 7.1 输出要求

在固定的两个完整 val 场景上生成：

- 六视角原图、2D 预测框、3D 框投影，明确模块和类别命名空间。
- LiDAR BEV、各 3D 模型预测与可选 GT；雷达位置/RCS/速度图。
- 分割图与完整图例；深度图固定米制色条和稀疏误差图，避免逐图自动缩放色条制造相似效果。
- MapTR 预测矢量与单独标识的 GT。
- 占用 PLY、BEV/高度切片、unknown/valid mask 图例。
- M05→跟踪器与 M07→跟踪器的两段轨迹对比。
- 包含全部任务面板的联合回放 MP4，旁附逐帧 sample_token、timestamp、源采样间隔和视频播放帧率。

结果缺失的面板显示具体 blocked/failed 状态，不填入上次缓存、真值或别的模型结果。这样的视频可以作为故障诊断产物，但不能判定里程碑 A 完成。

主回放保持完整连续场景。额外挑选有代表性的 bad case 图用于报告时，须保留固定基准场景和完整结果，不能只展示成功帧。

### 7.2 报告与表格

生成静态 reports/index.html，离线浏览即可查看指标、版本、图片和视频索引；不需要外网发布。生成 reports/final_report.md，包含：

1. 已完成的里程碑与模块状态；未完成项的具体原因和下一条命令。
2. 实际硬件、环境、模型版本、数据范围、标签和权重来源。
3. 各任务真实精度、样本数与协议；原生和导出后结果分列。
4. 单模型与整套回放实测时延；冷启动和 warm 性能分开。
5. 模态/模型横向对比、两种检测源的跟踪对比。
6. 训练实验 A/B 的假设、配置差异、真实曲线和结果。
7. 至少 8 个真实可复核的错误案例；不足 8 个时说明范围，不捏造。
8. 兼容性改动、仍存在的数值偏差、适用范围。
9. 已实际执行通过的复现命令和结果目录。

主表不预填任何 AP、mIoU、FPS 或收益数字。参考论文指标可单独存 paper_reference.csv，必须有来源、配置、数据协议、source=paper_reference，不参与实测验收。

## 8. 状态、验收与恢复

### 8.1 状态格式

每一步采用 pending/running/passed/failed/blocked/not_applicable 之一，not_applicable 只能用于计划本来未要求的子步骤，且带 reason。记录 started_at、finished_at、command、exit_code、input/config/code/weight hashes、artifact paths 和错误摘要。

每个模型至少分别跟踪 assets、environment、native_smoke、replay、evaluation、visualization、benchmark、export、export_validation；M11 用实际 CPU 关联路径记录运行方式。M13 另跟踪 train_A、train_B、evaluation。字段存在不代表通过。

一条有效指标至少有 value、unit、direction、protocol、split、scope、sample_count、run_id、source=measured、raw_evaluator_path。尚未执行的 value=null，原因明确。

### 8.2 必须具备的少量针对性校验

- 数据和几何：真实标定下的坐标变换、框尺寸/朝向、跨传感器 sample 关联。
- 适配层：同一原生预测适配前后类别、框、score 数量一致；数值差异有解释。
- 时序：场景切换清零；同一场景独立运行和批处理运行结果应在预定容差内一致。
- 导出：相同输入上的真实数值及任务指标回归。
- 产物：逐 token 覆盖、有限数值、数组 shape、所有报告资源可访问、JSON schema/CSV 字段、日志与哈希对应。

不要写大量只断言文件名或目录存在的测试。验收程序必须拒绝必需模块缺帧、GT 冒充预测、source 非 measured 的主指标、过期缓存和尚未运行的指标；真实性依赖完整运行记录与数据来源核查，不能声称一段脚本可自动证明所有结果都真实。

### 8.3 调试和阻塞处理

导入错误：先定位版本/API/编译问题，最小修复并记录 patch。
CUDA 算子错误：核对真实 GPU 架构、编译 flags、PyTorch ABI、SpConv/DCNv2/MMCV 版本和输入设备。
输出空或指标异常：核对权重完整加载、类别、预处理、单位、框中心、yaw、global 转换、评估范围和时间补偿；不要先改阈值掩盖根因。
OOM：确认 no_grad/eval、释放上一个模型、batch=1、原生允许的推理精度和分块方式。不得默默降分辨率、删相机、删 sweep 或换模型；任何改变输入协议的实验单独命名，不顶替基线。
导出不一致：先对齐预后处理，再定位关键算子/混合精度；不要跳过检查。
数据缺失：给官方入口和精确缺失路径，继续独立分支；关联评估保持 blocked。

每次恢复先读取 TASK_STATE、status 和日志末尾，验证资产再推进。无外部阻碍时持续执行，不在完成一个模块后反复问“是否继续”。确需用户补充账号、许可或文件时，一次汇总目前确认的缺失项和受影响模块。

## 9. 最终交接给用户

完成时直接给出：达到的里程碑、实际模型运行/评估/导出/训练状态、实测结果总表、总览 HTML、联合回放视频、实验报告、真实可复现命令，以及剩余失败项。

只完成部分时明确写 partial，并列出完成了什么、什么没完成及原因。不要只交目录、占位结果、伪指标或“建议用户自行运行”的说明后声称全流程已完成。
