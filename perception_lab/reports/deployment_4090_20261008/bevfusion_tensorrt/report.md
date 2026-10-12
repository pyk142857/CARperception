# CUDA-BEVFusion TensorRT FP16 实测

本次实际执行覆盖 scene-0061 39帧与官方 mini_val 81帧。阶段状态 **partial**；推理运行、mAP/NDS固定预算和完整几何诊断分别判断。partial表示固定指标预算已通过、几何/数值复核仍有失败项，未改写原comparison.status或任何诊断阈值。

| Split | 帧数 | mAP | NDS | ΔmAP | ΔNDS | forward含传输 P50/P95 ms | 配对状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| scene0061 | 39 | 0.709706 | 0.621329 | -0.002055 | -0.001018 | 10.568 / 12.126 | failed |
| mini_val | 81 | 0.576408 | 0.513238 | -0.002625 | -0.001233 | 10.036 / 11.497 | failed |

精度预算在首轮结果前固定：同split mAP、NDS 各下降不超过 **0.005**，未因结果调整。逐类AP见 [per_class_ap.csv](per_class_ap.csv)，几何/分数/速度分布与预先固定阈值见各split/comparison_summary.json。框按同类、中心距离≤2m的Hungarian匹配诊断；不等同官方检测评估匹配。

[完整指标CSV](metrics.csv)；各split的 predictions.json.gz/native_predictions.json.gz 保存实际标准化/原生输出，不筛除低分。timings.csv 为逐帧计时，performance.json 给出边界与显存；manifest.json 固定配置、engine、ONNX及native库哈希。FP32 CPU预处理与参考共用 pinned pipeline，向CUDA传FP16输入；传输差异不能称纯TensorRT kernel速度。

默认Myelin head 的失败尝试保留于 `attempts/myelin_head/`，其预测、指标与固定预算不覆盖。上游注明该融合可能有LayerNorm问题，另构建 `head.bbox.layernormplugin.plan` 进行定位；最终实际head以当前manifest配置/engine哈希为准，不能仅因采用插件便声称精度问题已解决。

补充的分数分层和五个实际部署差异案例见 [geometry_diagnosis.md](geometry_diagnosis.md)，逐类表格不改变完整配对验收范围。

范围是mini教学诊断，可能与预训练数据重叠，不是官方完整val；属性沿参考真实来源，不用GT补预测。桌面RTX4090时延不代表车端硬件。构建与来源见 [build_report.md](build_report.md)，权重审计见 [weight_audit.json](weight_audit.json)。
