# 4090 部署恢复发布与 FP16 补查（2026-10-12）

原实验报告与必要实现已恢复到 main `1f866fc8c493588b722902cdd0e503c1cd5daedb`，571 个文件路径已核对，其中 570 个与原完成提交字节一致，README 合并保留最新说明。本次补查完成，但 **FP16 全候选一致性仍未通过，未宣称精度修复或新的加速**。这是 nuScenes mini 教学诊断，不是全量验收。

## 发布与身份

来源提交 `ce244828fe5ffd73753ed672dfaf665f43502999` 保留；从远端 main/运输分支建立独立 worktree，先恢复小报告与代码，再分批运输必要较大文件，未强推。331 个缺失文件约 58.6 MiB 逐批读取远端原始字节核对 SHA256；全部 571 路径及 12 项核心远端内容再次验证。最新审阅、指令及 GPU 说明保留。见 [发布回执](publication_receipt.json)、[批次记录](publication_transport.json)、[完整清单](publication_inventory_verified.json)。后续诊断发布的独立核验也记入回执。

[运行前核验](preflight.json)检查原 manifest 329 项及 14 项大资源身份，无不一致；已有 39/81 帧结果复用。生产库、engine、预处理、阈值均未改。原始数据、权重、engine、RRD 和大型捕获留本地，路径与哈希保留在各实验记录。TRT runner 部分启动版本由精确编辑记录事后重建，未冒称启动时独立哈希；详见 [版本证据](trt/runner_version_provenance.json)。

## 运行、评估、诊断分别看

|项目|实际结论|
|---|---|
|BEVFusion reference/TRT 推理与官方代码 mini 评估|原 39/81 帧已完成，模型和输入身份核验后复用|
|mini_val 精度预算|reference mAP 0.5790331434 / NDS 0.5144708836；TRT 0.5764076609 / 0.5132377729，下降 0.00262548 / 0.00123311，满足各 0.005 预算|
|全候选中心、尺寸、yaw 一致性|原失败保留；本次未更改阈值或删低分预测|
|固定输入重复|4 帧，各后端 5 次未插桩生产运行及 5 次独立捕获运行；完整路径均观察到变异|
|同上游模块回放|4 帧×5 模块×2 上游，局部 camera/pool/view/fuser+decoder 通过固定张量容差；head 来源对齐后仍有少量超容差|
|pooling 候选|实验完成，无稳定提速，默认 upstream|
|Sparse4D|39/81 帧、原生 ID、场景重置及 DetectionEval/TrackingEval 摘要复核通过|
|GaussianWorld|39 帧推理及网格证据通过；缺精确匹配 SurroundOcc GT，评估 blocked_data|
|可视化|既有真实截图、RRD、四段视频和时间/token 对齐复核；本轮未重新运行浏览器 QA|

## FP16 差异实际定位

四个 token 为首帧 `ca9a282c9e77460f8360f564131a8af5`、高分异常帧 `747aa46b9a4641fe90db05d97db2acea`、`e0845f5322254dafadbbed75aaa07969`、`b26e791522294bec90f86fd72226e35c`。两侧 FP32 图像、点云和四个矩阵输入逐字节相同，TRT 之后采用原有 FP16 转换。

1. **同后端也有变异。** reference 最早变异出现在投影稀疏深度；实际 scatter 输入相同，改变像素均有多个点贡献，两个值都能在贡献中找到。TRT 稀疏深度及 LiDAR BEV 也有变异。不能将完整路径全部差异归因 FP16；也不能据此认定全部跨后端误差只是随机波动。[reference](reference/report.md)、[独立 A/A 复算](reference/production_aa_audit.md)、[TRT](trt/report.md)。
2. **几何差异独立存在。** 四帧贡献总数相同，但 126/104/136/156 个 BEV 格子的贡献分配不同（按上述 token 顺序）。此前排他末区间修复仍正确。用相同特征、深度、FP32 kernel 和 ray 顺序，只切换实际映射，8 组对照仍产生 25–140 / 10,368,000 个超容差元素，最大绝对差 0.108–0.227。未把边界量化差异武断归因某一算术操作。[对照结果](reference/replay/report.md)。
3. **候选匹配会夸大局部框差异。** `747...` 的 marked 诊断图按原同类 2m Hungarian 匹配差 1.6280m，配到了不同来源；真实同源中心差仅 0.01872m。`e084...` 出现局部峰值 mask 位置交换：源 181015 分数从 0.653751 变 0，邻源 180836 从 0 变 0.673828，不只是全局 TopK cutoff，也不是后置框 NMS。原始 pre-mask TRT logits 未捕获。[案例](trt/candidate_source_cases.json)。
4. **同源也可能有真实差异。** `b26...` 同源 yaw 差约 0.33761rad；给 head 相同 middle 输入时仅约 0.007411 / 0.000470rad。仅让 FP32 reference head 改用 TRT middle 已重现约 0.338rad 变化，支持上游输入的重要作用，尚未定位具体敏感层。[控制实验](trt/high_score_same_input_head_control.json)。
5. **捕获改变图的限制必须保留。** 新增 TopK 输出的 marked engine 与生产 head 六输出均非逐位一致；上述来源解释只归属于诊断图，不能倒推历史生产 query。相同 middle 的来源对齐 head 仍有少量回归残差，score 在固定容差内。只有首份模块原始数组归档，五次重复另保留统计，不能独立从五份数组重算。[独立审阅](trt/independent_review.md)。

浮点判定始终为 `abs(candidate-reference) <= 0.05 + 0.02*abs(reference)`，非有限值失败；整数索引精确相等。分位数、shape/dtype/layout、近零误差及超容差数在子报告 JSON/CSV 中。分数 ≥0.25/0.5 只用于解释，不改变验收输入。张量坐标容差不等于物理米制框容差。

P2F 是“有具体定位依据再改单层精度”。目前没有确认具体敏感层；已有 LayerNorm 的 FP32 累加亦经源码核验，不能算本轮新修复。因此未盲改 head 精度、未替换生产 engine，也未重复原 39/81 帧评估。本轮诊断运行不参与性能对比。下一步应对上游 middle 差异进一步分解、捕获 pre-mask 热图及精确几何边界，再决定有针对性的算子精度对照。

## Pooling、其他模型与可视化

90 行 kernel 与 30 行全模型配对复算：候选各组 kernel 更慢，全模型平均节省 **−0.035988ms**，仅 13/30 对更快且顺序敏感。默认 upstream，`experiment_completed=true / stable_speedup=false`。本次相同输入 pool 重复逐位稳定，完整路径上游却变化，说明“pool 相同”不保证“整条路径相同”；未追溯性声称每个旧差异都有唯一根因。没有新增盲目 block 扫描；Nsight Compute 仍因 ERR_NVGPUCTRPERM 未测，不声称带宽/occupancy 瓶颈，不混用边界算 Amdahl。[复算](pooling/recomputed.json)。

Sparse4D mini_val AMOTA 0.6375665930、AMOTP 0.8195541182、IDS 16；scene AMOTA 0.6794865962、IDS 9。GaussianWorld 保留原 25,600 Gaussians、原网格与 scene-stream；扫描可访问本地数据目录后，39 token 无精确 GT 匹配（一个 lost+found 无权限，不声称全机绝对不存在）。未用 LiDARSeg/伪标签代替。旧导出源码 hash 差异恢复为仅空白行四空格差别，恢复文件 SHA 精确命中原记录、AST 相同，未篡改原 provenance。[P4 审计](p4_audit/report.md)。

27 张真实图解码、4 段视频完整解码均通过（各 39 帧，2fps 为播放速度）；原 RRD hash 一致。原首/中/末帧时间/token 对齐证据复核。轨迹逐帧 Clear、场景变化清空、保留末 12 次观测；同场景消失后再出现的 ID 仍可能连接旧轨迹，无时间间隔截断，明确保留此展示限制。

[原实验报告](../deployment_4090_20261008/report.md) · [状态](status.json) · [指标](metrics.csv) · [复现](run_commands.md)
