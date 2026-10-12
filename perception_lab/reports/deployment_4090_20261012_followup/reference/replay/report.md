# 相同输入模块回放与几何映射隔离

4 帧 × 5 个模块 × reference/TRT 两组上游，共 **40 次参考回放**均完成，退出码 0。每一对参考与 TRT 模块读取相同 NPZ 字节，浮点输入统一为 FP16 可表示值；参考计算保持 FP32，TRT 使用原生产 engine。该实验不计量性能，不改变模型生产配置。

[summary.json](summary.json) 保存输入、两侧输出的 SHA256、误差分位数、非有限值及超容差计数；[metrics.csv](metrics.csv) 汇总 144 行诊断。公式保持 `abs(candidate-reference) <= 0.05 + 0.02*abs(reference)`。

## 局部计算

- **camera backbone/depth、BEV pooling、view transform、fuser+decoder**：4 帧两组上游的全部可比输出均满足固定逐元素容差。
- pooling 的同输入比较固定同一个已修复 TRT 几何映射；只用 interval 实际有效区间，跳过无效 geometry 前缀。最后 interval 的排他终点仍等于 indices 数量，未重新引入旧末区间问题。
- **head**：行对行误差较大，但两侧 TopK 顺序/集合可能变化。按实际种子来源对齐 marked TRT 诊断图之后，仍有少量回归字段超容差，因此不能把全部 head 差异归结为排序。

真实来源对齐见 [source_aligned_head.json](source_aligned_head.json)。8 组 head 中公共来源为 199 或 200；共有 3 组交换了一个来源。来源对齐之后，每组各字段超容差数量范围：reg 0–3，height 0–4，dim 0–10，rot 0–4，vel 0–5，score 全部为 0。reg 是 head 特征图坐标，不直接当作米。

**marked 图六个输出均不与 production 逐位相同**。所以来源对齐只归属于 marked 诊断图；即使数值接近，也不能严称为生产 engine 的候选来源。本次定位到 head 局部计算仍有残差，尚未定位为某个具体敏感算子，也未据此实施全 head FP32 或重新构建生产 engine。

## 只改变几何映射的对照

额外完成 4 帧 × 两组相同 features/depth，共 **8 个映射对照**。使用同一组 FP16 可表示 features/depth、同一个参考 FP32 pooling kernel，并让两种映射的贡献均按原始 ray 源索引升序排列，排除不同输入、不同 kernel 和来源排序的混淆。

参考 geometry 由固定预处理矩阵和实际 `get_geometry` 重建，量化后的有效整数坐标全部与原先捕获的真实 `camera.pool.coords` 逐位相同。TRT 映射来自生产捕获的有效 intervals/indices。两侧贡献总数相同。改变映射后，pooling 输出仍有少量元素超过原固定容差：

|token|reference 上游：超容差个数 / 最大绝对差|TRT 上游：超容差个数 / 最大绝对差|
|---|---:|---:|
|ca9a282c9e77460f8360f564131a8af5|26 / 0.177874|26 / 0.183353|
|747aa46b9a4641fe90db05d97db2acea|25 / 0.108629|25 / 0.108456|
|e0845f5322254dafadbbed75aaa07969|96 / 0.226826|98 / 0.226386|
|b26e791522294bec90f86fd72226e35c|140 / 0.197775|134 / 0.180857|

每个 pool 张量有 10,368,000 个元素，P95/P99 误差均为 0，不代表所有元素相同。映射边界的少量贡献差异已经被独立量化；这与已修复末区间问题不同，也不是 FP16 累加误差的证据。排序控制输出逐位相同，详见各 case 的 `trt_map_summation_order_control`。

## 限制与复现

原全候选诊断失败状态、39/81 帧 mAP/NDS 预算和默认 pooling 实现不变。本次仅补充四个诊断帧，不替代官方全量评估。后续若做精度修复，须先定位 head 内具体敏感位置，再按原协议复核；不能仅凭局部容差通过声称完整路径通过。

实际 GPU 命令、模型和运行哈希见 [execution.json](execution.json)，日志见 [run.log](run.log)。原始共同输入、两侧输出与几何对照 NPZ 保留在 `runs/deployment_4090_20261012_followup/{reference,trt}/replay/`。CPU 汇总可在 `perception_lab/` 执行：

```bash
python3 reports/deployment_4090_20261012_followup/reference/replay/analyze_replay.py
```
