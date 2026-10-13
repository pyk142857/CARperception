# 跟踪算法与GPU算子优化候选调研

日期：2026-10-13。项目证据快照：[8774a8c](https://github.com/pyk142857/CARperception/blob/8774a8c9b35acc1ec31caee5d18c4113b0d349c9/README.md)。
状态：已核对论文/作者项目与适配条件，未安装或运行新候选。本报告是候选调研，不是部署验收，也不改变[已验证保留方案](../../RETAINED_ALGORITHMS.md)。

## 推荐选择

执行指令已发布：[20261013_03](../../instruction/20261013_03_frozen_bevfusion_tracking_gpu_optimization.md)，待本地执行。派发前进一步核查作者源码：OptiPMB原生需要真实LiDAR框内点数；GRAE原NMS先粗筛及三维IoU，再按分数过滤，且高度包含公式与物理标准公式存在差异。上述行为已进入执行指令，不能以空点云或二维NMS静默替代。

针对当前CARperception，优先开展两项独立工作：固定已有BEVFusion检测输出比较OptiPMB与原生GRAE；剖析GRAE的CPU几何计算/NMS，选择实际热点开发同语义GPU算子。IMM-MOT适合做运动模型模块对照。TileLang用于实现候选kernel，KernelBench用于正确性与性能评估，KernelAgent作为后续自动化优化框架。

下列适配优先级是结合论文、公开源码说明和本项目报告作出的工程判断，尚无本机新实测。主要候选为2025论文/项目；2026补充明确列年，不把较早预印本只因2025录用而改称2025首次提出。

## 跟踪候选

| 方法 | 年份/出处 | 主要改进 | 当前数据适配与定位 | 资源 |
|---|---|---|---|---|
| OptiPMB | 2025-03预印本；T-ITS 2025 | 使用随机有限集贝叶斯过滤，显式处理目标新生、存活与消失；测量驱动自适应新生、遮挡下检测概率调整、假设剪枝与轨迹提取 | 优先完整tracker对照。可适配现有global三维框、速度、分数、类别与时间戳；跟踪部分不需训练新网络。作者入口默认trainval/test，需适配mini读取、场景起始和结果导出。关联依赖FastMurty原生库 | [论文](https://arxiv.org/abs/2503.12968)、[官方代码](https://github.com/dinggh0817/OptiPMB) |
| IMM-MOT | 2025-02预印本；IROS 2025；2025-07源码发布 | 交互多模型过滤器融合不同运动假设；Damping Window根据持续匹配情况管理轨迹；Distance-Based Score Enhancement按距离调整分数 | 可复用已有三维检测结果，优先隔离验证运动模型、生命周期和分数调整三个模块。用于检验变速/转弯相关断轨；避免一次叠加全部模块后无法定位收益 | [论文](https://arxiv.org/abs/2502.09672)、[官方代码](https://github.com/Ap01lo/IMM-MOT) |
| TrackTrack | CVPR 2025 | Track-Perspective-Based Association从轨迹视角选取匹配检测；Track-Aware Initialization考虑已有轨迹来初始化新轨迹 | 二维MOT方法，官方项目包含YOLOX与FastReID，验证于MOT17/MOT20/DanceTrack。适合借鉴重复新生抑制与关联逻辑，或作为二维跟踪扩展；不能直接替换nuScenes三维关联后沿用官方AMOTA | [论文](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)、[官方代码](https://github.com/kamkyu94/TrackTrack) |

OptiPMB作者报告CenterPoint输入、nuScenes完整val AMOTA0.741、IDS227。这只是论文/仓库成绩，不能与当前mini结果直接横比，也不保证对BEVFusion检测输入有效。

IMM-MOT尤其需要同时观察IDS/FP/FN：作者消融表从基础方案AMOTA73.08、IDS232，到三个模块组合AMOTA73.79、IDS326；模型分数提高没有带来所有身份指标改善。其官方实现标注DBSE位于pre_processing/dbss.py、IMM位于motion_module/kalman_filter.py、DW位于tracking/nusc_score_manage.py，具体仓库目录以检出版本为准。这些模块名分别表示距离分数增强、交互多模型过滤、阻尼窗口生命周期管理。

OptiPMB中的“存在概率”表示某条轨迹仍对应真实目标的概率；随机有限集表示目标集合与数量都可变化。它提供与固定阈值/固定max_age不同的生命周期建模方案，并非给既有贪心算法增加一个常量。

## GPU优化项目与论文

| 项目/资料 | 年份 | 可复用内容 | 对本项目的作用与限制 |
|---|---|---|---|
| [TileLang](https://github.com/tile-ai/tilelang) / [论文](https://arxiv.org/abs/2504.17577) | 2025-01开源、2025-04预印本 | 用Python风格的分块编程表达GPU算子，支持布局控制、融合、归约及编译；当前CUDA后端覆盖SM70–SM120 | 适合为批量几何计算、门控与距离矩阵、点云scatter/分段归约实现候选。4090选择SM89适用路径，Hopper/Blackwell专用指令不能直接照搬；独立环境适配，不升级现有旧模型环境 |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) / [ICML 2025论文](https://proceedings.mlr.press/v267/ouyang25a.html) | 2025 | 论文包含250个PyTorch任务；分别检查功能正确和提速；当前工具支持本地CUDA/Triton/TileLang等后端评价 | 是评价工具而非现成感知算子。可借鉴其输入随机化、参考实现比较、计时与fast_p（正确且超过指定加速阈值的任务比例）来评估Codex生成kernel；需补本项目真实形状、阈值边界和整网核验 |
| [KernelAgent](https://github.com/meta-pytorch/KernelAgent) / [PyTorch硬件优化报告](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/) | 硬件反馈优化报告2026-03 | Nsight Compute分析→瓶颈诊断→改写Triton→数值验证→CUDA event计时→保留最佳版本；保存中间产物 | 适合将单个已定义参考语义的kernel交给代码agent迭代；需要可用的GPU profiling和LLM服务。官方H100收益不是4090收益，需使用实际硬件参数。框架验收不替代感知指标评价 |
| [NVIDIA BEVPoolV3优化说明](https://developer.nvidia.com/blog/accelerating-bev-pooling-on-nvidia-gpus-for-physical-ai-applications/) | 2026-06 | 分离五组INT32索引数组、预计算索引、减少重复depth读取、每个输出区间由单个owner累加并写回；根据工作集是否进入L2选择优化方式 | 与现有BEV pooling最相关。此次核对到优化说明和伪代码，完整V3插件源码尚未确认；可参考其设计改写已有CUDA-BEVFusion算子，不把它标为已下载即可部署的模型 |

BEVPoolV3的宣传加速比需要匹配基线和硬件：官方Blackwell同FP16比较V2+DO（CUDA-BEVFusion式depth-outer）37.8µs→V3 17.3µs，约2.18倍；相对更早V2的更大倍率不等于我们相对现有上游的收益。按实际rank/interval数组、depth、图像特征与输出张量大小判断4090的工作集，不能根据官方49MB配置直接断定本项目进入L2。先研究FP16/FP32路径的索引与访存，FP8作为后续独立精度实验。

## 最值得做的三个算子任务

| 顺序 | 任务 | 具体工作 | 保持的语义与结果 |
|---|---|---|---|
| 1 | GRAE几何计算/NMS | 先拆CPU读取、筛选、排序、旋转框交并计算与抑制；批量GPU计算重叠，必要时用bitmask压缩成对抑制关系 | 原作者用什么IoU/GIoU、是否包含高度、类别规则、排序及同分选择都先核对。旋转BEV IoU不自动等价于三维IoU；普通二维TensorRT NMS也不能直接替换 |
| 2 | 关联输入与代价计算融合 | 合并坐标/速度回推、类别门控、二维距离与无效配对屏蔽，减少临时数组及小kernel调用；适用于需要GPU关联特征的链路 | 距离和门限、float32转换位置、候选次序保持；带数据搬运比较CPU/NumPy与GPU端到端。当前0.772ms/帧CPU候选已经较快，收益需要重新证明 |
| 3 | BEV pooling分段归约 | 真实输入上尝试索引分离、预计算与区间归属写回，再调channel tile、寄存器和访存 | 几何区间和累加精度保持可核验；同时检查输出张量和官方检测指标。上游已具备depth-outer，不能把已有技巧再次算作新增优化 |

现有[GRAE报告](https://github.com/pyk142857/CARperception/blob/8774a8c9b35acc1ec31caee5d18c4113b0d349c9/perception_lab/reports/perception_2025_20261013/grae3dmot/report.md)记录固定BEVFusion输入的CPU预处理/NMS阶段约139.56/240.84ms每帧（39/81范围），因此建议先剖析这一段。上述总时延不是单独IoU kernel时间，需分段确定可GPU化比例；它也不是新Mamba输入的性能测量。

## 验证顺序

1. 跟踪方法对照固定BEVFusion预测文件和同一39/81帧split，检测器保持不变。OptiPMB与GRAE分别使用作者原生入口与可追溯mini适配；用官方TrackingEval比较AMOTA、AMOTP、IDS、FP、FN，另保留失败事件。不通过提高分数阈值减少输出来冒充关联改善。
2. IMM-MOT先完整原生运行确认可复现，再隔离IMM/DW/DBSE，明确检测分数调整属于不同实验，不与“固定全部输入分数”的纯关联对照混合。
3. GPU任务先验证真实形状、空输入、相等代价、边界和原算子输出；记录H2D/D2H，再进行预热后的交替顺序配对。只有局部计时与整网结果均有改善时列为成功优化。
4. CUDA/Nsight数据按实测报告；无权限或未取得计数器时保留CUDA events等实际信息，不编造带宽、occupancy或roofline结果。KernelBench只是kernel层评价，最终仍需实际模型及官方指标检查。

本轮只完成资源调研和方案建议，没有启动新的下载、训练、跟踪或GPU性能实验。模型替换与算子提速分开验收；当前已通过的保留方案继续按其原报告执行。

## 来源索引

1. [OptiPMB论文，arXiv:2503.12968](https://arxiv.org/abs/2503.12968)；[作者代码与数据/关联依赖说明](https://github.com/dinggh0817/OptiPMB)。
2. [IMM-MOT论文，arXiv:2502.09672](https://arxiv.org/abs/2502.09672)；[作者代码、消融表与模块入口](https://github.com/Ap01lo/IMM-MOT)。
3. [Focusing on Tracks for Online Multi-Object Tracking，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html)；[TrackTrack源码](https://github.com/kamkyu94/TrackTrack)。
4. [TileLang论文，arXiv:2504.17577](https://arxiv.org/abs/2504.17577)；[源码及当前硬件支持说明](https://github.com/tile-ai/tilelang)。
5. [KernelBench，ICML 2025](https://proceedings.mlr.press/v267/ouyang25a.html)；[源码与本地计时评价](https://github.com/ScalingIntelligence/KernelBench)。
6. [KernelAgent硬件反馈优化，PyTorch 2026-03-06](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)；[代码](https://github.com/meta-pytorch/KernelAgent)。
7. [BEV pooling优化，NVIDIA 2026-06-24](https://developer.nvidia.com/blog/accelerating-bev-pooling-on-nvidia-gpus-for-physical-ai-applications/)。
