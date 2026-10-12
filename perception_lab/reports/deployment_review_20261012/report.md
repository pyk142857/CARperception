# 4090 部署 partial 验收审阅

日期：2026-10-12。审阅范围：GitHub 可读取的原始报告、JSON/CSV 与目录；本次未在工作站运行 GPU 实验。

**结论：保留 partial。** BEV pooling 候选实验已有正确性与计时证据，保持上游默认合理；TensorRT 的任务指标预算通过，但几何与中间张量一致性尚未全部通过。总报告、实现代码及 Sparse4D/GaussianWorld/同步回放的核心证据尚未完整上传，不能将本次审阅写成全流程独立复核通过。

## 1. 远端状态与证据范围

审阅起始 main：e9039359b8dfc331fcdaed09ee0c4c25100fed4c。BOSS 提供的本地完成提交 ce24482 在 GitHub commits API 中尚不可解析。

临时分支 codex/deployment-upload-ce24482 的已读取 tip 为 51d31c44838457e67618dd3c8804479a28559d80，tree 为 cb78cd8f14ebf939847640919a5c21722b09b7cb。该分支相对审阅起始 main ahead 24 / behind 0，但只完成部分运输，不能将其视为完整发布分支。

原 artifacts_manifest.json 列出329个报告目录产物，相同路径在临时分支树中存在159个，缺170个。本数字只检查路径存在，不表示159个文件已全部逐字或逐哈希验证；清单未覆盖全部代码及总报告。详见 publication_inventory.json。

目前 main 和临时分支都没有原实验总 report.md；status.json、总 metrics.csv、run_commands.md、修复后的 pooling 中间张量诊断、Sparse4D 指标、GaussianWorld 与同步可视化关键证据也未全部上传。读取范围没有扩展到 BOSS 工作站的 /home/minglei/Desktop/CAR 或 localhost:9094。

| 模块 | BOSS 提供的本地状态 | 本次可独立读取的证据 |
|---|---|---|
| BEVFusion reference | 39帧＋81帧推理/评估完成 | 子报告、指标、坐标审计及部分原始产物 |
| TensorRT FP16 | 指标预算通过，几何/中间张量未全部通过 | mini_val比较摘要、几何尾部诊断、边界修复记录、构建记录及指标CSV |
| BEV pooling | 候选正确，无稳定提速，上游默认 | 报告、90轮算子配对CSV、30轮整模型配对CSV、profiling摘要 |
| Sparse4D | 39帧＋81帧推理/评估完成 | 核心结果尚未上传，待复核 |
| GaussianWorld | 39帧推理完成，缺占用GT | 核心结果尚未上传，待复核 |
| 同步展示 | 四模型首/中/末检查通过 | 核心证据尚未上传，待复核 |

## 2. TensorRT：平均指标达标，一致性仍待闭合

| split | 帧数 | reference mAP / NDS | FP16 mAP / NDS | mAP / NDS下降（百分点） |
|---|---:|---:|---:|---:|
| scene0061 | 39 | 0.711761 / 0.622348 | 0.709706 / 0.621329 | 0.2055 / 0.1018 |
| mini_val | 81 | 0.579033 / 0.514471 | 0.576408 / 0.513238 | 0.2625 / 0.1233 |

两项均满足预先固定的0.005（0–1标度，即0.5个百分点）预算。mini_val的bicycle、traffic_cone AP差值分别约−1.000、−1.556个百分点，平均指标不能代替逐类检查。

mini_val全候选匹配的中心/尺寸/yaw差异P95为0.7203m / 0.3006m / 0.4429rad，分别超过既定0.2m / 0.2m / 0.1rad诊断阈值。该比较使用同类2m Hungarian配对，是两后端输出之间的诊断，不是官方AP匹配或GT定位误差。

分组诊断解释了尾部构成，但不能用于重新定义验收：
- 双方分数≥0.25的3176对：中心P95 0.0512m，尺寸0.0287m，yaw 0.0730rad；分数差P95仍约0.0330，超过0.03诊断阈值。
- 中心差>0.2m的2333对中，2328对至少一侧分数<0.25（99.79%）。
- 双方分数≥0.5仍有2对中心差>0.2m、37对yaw差>0.1rad。
- 例：mini_val token 747aa46b9a4641fe90db05d97db2acea 的pedestrian配对，分数0.5136/0.6372，中心差1.6286m。

已有诊断明确本配置不执行后置NMS。邻近候选、TopK选择和Hungarian配对交换是待验证解释，不能把问题归因于NMS，也不能把全配对差异全部当成FP16数值误差。下一步需要同后端重复基线、原生候选来源索引，以及相同层输入下的逐层比较；query种子已改变时不能直接按输出行号对齐。

中间张量失败信息目前来自BOSS的本地摘要。修复后 pool_intermediate_audit.md / pool_intermediate_frame_0.json 尚未上传，本次没有替它们给出根因。

## 3. 有具体证据的 CUDA 正确性修复

geometry_bounds_fix.md 记录：collect_starts_kernel 最后区间终点原为 numel_geometry−1，下游按 i<end 访问，导致遗漏最后一个贡献；修复为排他终点 numel_geometry。

真实首帧累计贡献由1,835,477补为1,835,478。三种直接运行生产kernel的边界测试，修复前失败、修复后返回0。当前已读取说明与测试JSON，补丁及测试源代码仍待上传后核对。

这是算子正确性修复，不是提速成果；它只能解释已定位的遗漏，不能证明其余FP16误差已全部消除。后续部署与捕获应使用同一修复版本和对应库哈希。

## 4. BEV pooling：实验已完成，候选不作为默认

候选每线程处理4通道，block=(20,8)；上游每线程10通道，block=(8,128)。三份真实输入的候选/上游输出逐位一致，边界与Compute Sanitizer记录通过。

本次从原始90行CSV独立复算：

| capture | 配对轮数 | upstream P50 ms | candidate P50 ms | 配对加速比中位数 |
|---|---:|---:|---:|---:|
| frame_0 | 30 | 0.224656 | 0.247808 | 0.900 |
| frame_19 | 30 | 0.212896 | 0.247808 | 0.862 |
| frame_38 | 30 | 0.217088 | 0.250032 | 0.868 |

三个输入均变慢。30轮常驻完整路径原始CSV复算：上游均值8.466408ms，候选8.502396ms；平均节省−0.035988ms，仅13/30对候选更快。按顺序分组的平均节省为+0.239320/−0.311297ms，顺序影响明显，不能挑选有利顺序宣称收益。

Nsight Systems记录block数761→12,166，寄存器40→38；更小block、更多channel worker可能增加调度与重复索引/depth读取，但这里只能写作解释假设。Nsight Compute因ERR_NVGPUCTRPERM失败，未获得带宽、stall、achieved occupancy等硬件计数器，不能宣称已证明memory-bound。

算子输出逐位一致，但完整模型输出和指标存在小差异；其他阶段的同后端重复变异尚未单独隔离。建议补A/A基线用于解释，不因此推翻已保存的候选正确性结果。

## 5. 时延的使用边界

mini_val报告的reference端到端P50为269.513ms，TensorRT为143.792ms。各自forward定义不同，不能把134.695ms与10.036ms直接当作严格同边界的纯GPU加速比。

pooling约0.22ms，说明其绝对时间较小，但当前分段计时和整模型计时边界不同；若计算Amdahl上界，必须补同一未插桩关键路径下的占比，不能用不同计时相除得到正式收益上界。

以上均是4090桌面GPU、mini教学范围结果，不代表车端实时闭环或官方全量val。

## 6. 后续工作与状态

1. 恢复完整可审阅发布，优先小体积总报告、状态、指标、代码/补丁/测试、修复后中间张量证据及各模型摘要；再运输必要大文件。保留ce24482，不覆盖已发布文档。
2. 复核同后端重复、TopK来源和逐层张量；必要时仅对确认敏感的层做FP32对照。固定阈值与官方评估输入保持一致。
3. pooling标为experiment_completed=true、stable_speedup=false、selected_default=upstream。没有证据支持继续盲目调block。
4. GaussianWorld按原指令分开记录inference passed / evaluation blocked_data；有精确token匹配的官方占用GT才计算IoU。缺GT不影响已完成推理的归档。
5. 发布后复核Sparse4D官方检测/跟踪指标、场景重置和四模型同步截图/时间轴。当前摘要不能替代这些文件。

执行指令：[20261012_01](../../instruction/20261012_01_deployment_publish_and_fp16_followup.md)。

## 7. 固定版本证据入口

- [原manifest](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/artifacts_manifest.json)
- [BEVFusion reference子报告](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bevfusion_reference/report.md)
- [TensorRT metrics.csv](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bevfusion_tensorrt/metrics.csv)
- [mini_val比较摘要](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bevfusion_tensorrt/mini_val/comparison_summary.json)
- [几何诊断](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bevfusion_tensorrt/geometry_diagnosis.md) / [JSON](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bevfusion_tensorrt/geometry_diagnosis.json)
- [geometry修复说明](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bevfusion_tensorrt/geometry_bounds_fix.md) / [测试记录](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bevfusion_tensorrt/geometry_bounds_test.json)
- [pooling报告](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bev_pooling/report.md)
- [90轮kernel原始CSV](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bev_pooling/paired_timings.csv)
- [30轮完整路径CSV](https://github.com/pyk142857/CARperception/blob/51d31c44838457e67618dd3c8804479a28559d80/perception_lab/reports/deployment_4090_20261008/bev_pooling/full_model_paired_timings.csv)

本审阅保留论文/源码候选、实测结果与缺失证据的区别。后续远端状态变化后，按新commit重新核对。
