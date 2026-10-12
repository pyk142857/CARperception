# Reference 同后端重复诊断

四个固定帧各完成 3 次 warmup、5 次未插桩生产 forward；随后对同一个已预处理 batch 各完成 3 次 warmup、5 次插桩 forward。共 20 次生产输出与 20 组、每组 52 个捕获张量。配置、权重、相机顺序、实际 sweeps 与输入身份见 [identity.json](identity.json)、[execution.json](execution.json) 和各 token JSON。未进行官方全量评估，也未计量性能。

四帧未插桩生产输出均非逐位稳定，见 [production_summary.json](production_summary.json)。这项检查只比较完整输出序列，未把行号当作候选身份。插桩捕获每次实际 `argsort` 的输入分数、源索引、排序间隔，并通过原解码的分数/范围 mask 将框映射到 `class*HW+spatial_index`。来源集合在本次各帧重复中不变，但后三帧四次对比全部发生来源顺序变化。因此直接行对行比较会产生并非同一候选的巨大差异。

## 实测的首个变化位置

[稀疏深度碰撞证据](raster_collision_evidence.json) 覆盖 4 帧 × 4 次与 repeat0 的对比，共 16 对。在全部对比中，实际 scatter 的行坐标、列坐标和待写深度值逐位相同。所有发生变化的 sparse-depth 像素均有多个投影贡献，且两次输出都属于该像素的实际贡献值集合。LiDAR BEV 在这 16 对中逐位相同。

|token|每次对比变化的深度像素范围|深度最大绝对差|实际来源对齐中心最大差 m|普通 2π 环绕 yaw 最大差 rad|
|---|---:|---:|---:|---:|
|ca9a282c9e77460f8360f564131a8af5|0–2|0.370426|0.004957|0.001119|
|747aa46b9a4641fe90db05d97db2acea|487–631|38.032069|0.070766|0.016573|
|e0845f5322254dafadbbed75aaa07969|1288–1380|45.494144|0.072741|0.021372|
|b26e791522294bec90f86fd72226e35c|848–1519|56.708123|0.509560|0.031764|

这直接证明本次参考实现的多点同像素深度写入存在重复运行不确定性。相机 feature/depth weights、BEV pooling 和后续 head 随之变化；该发现不等于全部 FP32/TRT 差异已经定位或修复。插桩加入同步，可能改变竞争时序；生产输出与插桩输出分开保存，不能根据某次捕获稳定就宣称生产稳定。

## 指标解释与边界

详细字段统计见 [metrics.csv](metrics.csv) 与 [source_aligned_metrics.json](source_aligned_metrics.json)。原始 native yaw 在 ±π 处可能相差约 2π，`summary.json` 的 `source_aligned_native_max_abs` 是各字段原生数值的最大绝对差，**不能解释成物理角度或中心差**。本页按字段分开统计，仅作普通 2π 环绕；未使用 barrier 的 π 对称性改变验收。

浮点张量判断式为 `abs(candidate-reference) <= 0.05 + 0.02*abs(reference)`；非有限值失败。整数来源、坐标、标签必须严格相等，不使用相对误差放宽身份判定。各 token JSON 保存 shape/dtype/layout、NaN/Inf、哈希、误差分位数、超容差数量，以及参考绝对值 ≤0.01 时的独立绝对误差。接近零的元素不靠相对误差单独作结论。

`summary.json` 中的原始 `head.*` 通配说明仅指 head 的 BCQ 回归张量；具体节点 `tensor_stats.layout` 为准，dense heatmap 是 NCHW、head input 是 NCHW、pool coordinates 是 Nx4 整数 xyz/batch。汇总中的容差表述对浮点适用，整数按各节点记录严格相等。

原固定全候选阈值与官方 mAP/NDS 预算不变；本次结果不能替代原 39/81 帧评估。未插桩候选来源不可用，生产端 Hungarian 只用于既有输出诊断，不构成真实候选身份。

## 产物与复现

每个 `<token>.production.json` 保存 5 次未插桩 native/standardized 框；`<token>.json` 保存 5 次捕获的同类结果、真实来源与张量统计。全部原始 NPZ 留在 `runs/deployment_4090_20261012_followup/reference/`，约 5.6 GiB，不进入 Git。完整日志见 [run.log](run.log)。

[复现命令](run_commands.md) 包含 GPU 运行和 CPU 证据复算。CPU 分析脚本为 [analyze_raster_collisions.py](analyze_raster_collisions.py)。模块同输入回放在 TensorRT 生成共享输入后另行运行，结果独立记录。

模块回放现已完成：见 [40 个模块回放与 8 个映射对照](replay/report.md)。camera/pool/view/fuser 的共同输入比较通过固定张量容差；几何映射本身产生少量超容差元素，head 按真实来源对齐诊断图后仍有局部残差，完整差异尚未闭合。

独立 CPU 审计见 [production_aa_audit.md](production_aa_audit.md)：40 个原始输出及标准化重算通过，覆盖组内全部配对及生产/插桩交叉配对，保留原同类 2m Hungarian 输出诊断与分数组解释。
