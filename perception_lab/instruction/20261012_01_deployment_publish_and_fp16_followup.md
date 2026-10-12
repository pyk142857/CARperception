# Instruction 20261012_01：恢复部署发布并补齐 FP16 一致性诊断

## 目标与已有依据

Codex：继续CARperception的20261008_01部署任务。先让BOSS能从GitHub读到完整报告与必要实现，再定位尚未闭合的TensorRT FP16差异。按原计划补全已有任务，不把无提速或缺标签改写成全部passed。

先读AGENTS.md、README、TASK_STATE、原20261008_01，以及 reports/deployment_review_20261012/report.md / publication_inventory.json。原执行目录为 /home/minglei/Desktop/CAR；若实际路径不同，从现有仓库定位，不创建空目录冒充已有工程。

BOSS提供的本地完成提交为ce24482。本次审阅起始远端main为e9039359b8dfc331fcdaed09ee0c4c25100fed4c；临时运输分支codex/deployment-upload-ce24482的已审阅tip为51d31c44838457e67618dd3c8804479a28559d80。执行时读取最新远端，不假定这些仍是head；本指令及审阅报告会产生新的文档提交。

本地已完成39帧/81帧的结果优先复用；仅在身份哈希不符、缺失结果或确定诊断需要时重跑对应部分。新增诊断放在 reports/deployment_4090_20261012_followup/，原结果与失败记录保留。

## P0：先检查真实工作状态

- git status；解析ce24482的完整commit，记录当前分支/HEAD、origin/main与运输分支head。保留已有未提交工作与完成提交。
- 读取本地总report.md/status.json/metrics.csv/run_commands.md及修复后pool_intermediate_audit.md。逐项对应真实代码、库/engine/输入哈希、39/81帧覆盖，不能只根据本摘要生成缺失结果。
- 对已完成阶段分别记录推理、评估、一致性、提速、可视化、发布状态，避免同一个passed涵盖所有性质。

## P1：恢复远端发布，先运输可审阅的小文件

优先发布原 reports/deployment_4090_20261008/ 的总report、status、metrics、run_commands、manifest；各模型摘要/官方metrics_summary；修复后的中间张量诊断；S4D/GaussianWorld/可视化摘要和代表帧；部署配置与实际tools/tests/patches。先使报告对应的入口代码可读，再完成必要的较大产物。

1. 以最新远端运输分支或main为起点，在独立worktree建立发布分支；复用远端已经存在的对象。保留本地ce24482作源记录，不从包含全部未上传历史的ce24482直接新建运输分支后再次推整个大pack。
2. 依据 publication_inventory.json 的缺项与本地git diff构造待运输列表。该清单仅覆盖原manifest，不包含全部代码和总report.md；额外核对真实变更文件。
3. 以路径为单位、按新对象总量分批提交/推送。建议每批不超过5MiB，单独处理更大的必要文件；先小报告与代码。每批读取远端文件/树并核对原始内容SHA256后再继续，记录运输原commit与新commit映射。
4. 使用命令级HTTP/1.1等设置时记录实际差异，不全局覆盖网络配置。小批push仍失败，可用现有GitHub认证，通过Contents/Git Data API按文件/blob和小tree上传，验证后再更新ref；认证内容不写入报告或日志。
5. 原始数据、权重、环境、engine、RRD、巨大capture按现有规则留本地；原预测/比较大文件可按既有gzip方式运输。压缩文件同时记录压缩字节哈希与解压原内容哈希，README/manifest链接必须指向真实可读路径。不要静默删除已承诺的小体积核心证据。
6. 运输分支文件未齐前不合入main。合入前与最新main核对并解决冲突，保留GPU原理报告、本次指令与审阅文档，更新README/TASK_STATE的实际状态。main更新必须是保留双方工作的普通合并或fast-forward，不强推覆盖。
7. 验证main上的总报告、状态、指标、核心代码、修复后张量诊断、S4D/GW摘要和代表图。最后再报告“已发布”；若网络仍不可用，给出已上传/缺失清单和精确失败日志，保留发布blocked，不将对象上传量等同main完成。

## P2：FP16诊断，从真实差异定位

已知mini_val mAP/NDS下降0.002625/0.001233，满足0.005预算；全候选中心/尺寸/yaw诊断失败。大部分中心尾部涉及低分，但双方≥0.5仍有2对中心异常与37对yaw异常。当前配置不执行后置NMS；不要再将NMS作为未经核对的解释。

A. **核对边界与身份。** 所有诊断使用相同已修复geometry、确定的预处理/相机顺序、实际点云sweeps、权重、ONNX/engine、插件与解码配置。pooling末区间排他终点修复已有记录，先核对补丁和生产kernel测试源码，不重复覆盖修复或使用旧capture。

B. **同后端重复基线。** 用固定的首帧和现有高分异常帧，reference/reference与TRT/TRT各做至少5次相同输入运行；记录原生张量、候选来源、最终输出变异。保留生产配置；若为了定位另做确定性设置，独立标注，不冒充原性能结果。

C. **读取修复后的中间张量失败。** 报告每个节点的shape/dtype/layout、NaN/Inf、绝对/相对误差分位数与超容差数量，明确tensor_atol=0.05 / tensor_rtol=0.02的实际判定式。不要只给allclose布尔值；接近零的相对误差单独解释。

D. **从第一个失配节点向下隔离。** 在实际可导出位置，按camera feature/depth、geometry映射与interval贡献数、BEV pooling、view transform、LiDAR BEV、融合、head的顺序比较；缺少可捕获节点就记录缺口。通过给两实现输入相同上游张量做模块回放，区分局部计算差异与上游误差传播。

E. **候选身份诊断。** 保存TopK前的同格点分数、被选中的源索引及排序间隔。若head query种子已变化，按真实来源对齐并记录候选交换，不直接把两列表行号当相同目标；同类2m Hungarian只保留为现有输出诊断。优先复核token 747aa46b9a4641fe90db05d97db2acea，以及geometry_failure_cases.json中的高分异常。框中心/yaw/尺寸/速度的原生与标准化差异分别核对。

F. **有定位依据再做精度对照。** 若具体层/插件被确认敏感，只对该处测试FP32输入/计算/累加或纠正layout/精度接口，其他路径保持。已有LayerNorm-plugin engine先检查当前实现，不重复宣称此前工作为新修复。修复后先重跑诊断帧，再重跑同一39帧与81帧官方评估及一致性。精度和计时边界分别报告。

全候选诊断阈值、原mAP/NDS预算、评估输入分数与官方代码保持不变。按≥0.25或≥0.5分组仅用于解释；不能删除低分预测或事后放宽阈值来改写passed。

## P3：收束 pooling 实验

90行kernel配对数据与30行整模型配对数据已复算：三组kernel均变慢；完整路径平均节省−0.035988ms且受测量顺序影响。保留candidate代码/正确性/负结果，默认upstream，记录experiment_completed=true、stable_speedup=false。

先按P2的同后端重复检查解释“pooling逐位相同而完整输出不同”的来源。现有证据不支持继续盲目扫block；只有新的profile显示可消除的明确成本才追加候选。Nsight Compute的ERR_NVGPUCTRPERM保持计数器未测状态，不自动修改驱动权限或宣称带宽/occupancy瓶颈。若计算Amdahl收益上界，使用同一真实关键路径计时占比。

## P4：复核Sparse4D、GaussianWorld与展示证据

- 上传并复核Sparse4D的mini_val DetectionEval/TrackingEval原始摘要、实际frame/token覆盖、场景重置、原生ID、AMOTA/AMOTP/IDS与测量边界。未测数值留空并标明原因，不依靠先前聊天摘要填写。
- GaussianWorld上传实际39帧grid/provenance/summary/audit/失败案例与代表图，保持原25,600 Gaussian、原输出网格与真实scene-stream协议。先检查本地或作者官方标签清单；存在精确token匹配的SurroundOcc GT时按原协议评估并报告覆盖；没有匹配GT则inference passed / evaluation blocked_data。LiDARSeg或伪标签不能代替占用真值。全量数据下载不作为恢复发布的前置条件。
- 展示保存首/中/末帧真实截图、时间轴/sample对应、占用轴序/类别图例、轨迹清理及视频解码覆盖证据。localhost链接是本机入口，GitHub另外给代表图与复现命令。

## 交付与验收

原实验完整发布后，后续诊断目录至少包含report.md、status.json、metrics.csv、publication_receipt.json、run_commands.md和新增诊断/案例。publication_receipt记录运输前后ref、来源ce24482完整SHA、最终main、核心文件SHA256与尚未发布的大产物位置；新诊断记录代码/输入/库/engine身份和计时范围。

最终分别报告：
- 已运行、已评估、通过/未通过的诊断；
- pooling是否有提速及默认实现；
- GaussianWorld推理与评估的独立状态；
- 远端main是否完整可审阅。

本指令已发布不等于已执行。闭环结论须以实际文件、测量与远端核对为依据。
