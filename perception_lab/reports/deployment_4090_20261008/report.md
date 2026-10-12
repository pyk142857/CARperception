# RTX 4090 感知部署实测报告

本轮使用 nuScenes mini：scene-0061 完整 39 帧和官方 mini_val 81 帧。所有预测来自本机模型真实推理；这些数据可能与预训练集重叠，不能作为全量泛化结论。

**当前验收状态：partial。** GaussianWorld 的实际推理与标签评估状态分别列于下方；缺失指标留空。各项限制以以下阶段报告为准。

|模型/后端|范围|mAP|NDS|模型边界 P50/P95 ms|端到端 P50/P95 ms|
|---|---|---:|---:|---:|---:|
|BEVFusion R50 / PyTorch|scene0061 (39)|0.7118|0.6223|122.3368 / 126.7673|244.2790 / 286.4772|
|BEVFusion R50 / PyTorch|mini_val (81)|0.5790|0.5145|134.6955 / 146.2021|269.5135 / 401.9760|
|BEVFusion R50 / TensorRT 8.6 + SpConv|scene0061 (39)|0.7097|0.6213|10.5684 / 12.1265|142.0893 / 216.1660|
|BEVFusion R50 / TensorRT 8.6 + SpConv|mini_val (81)|0.5764|0.5132|10.0356 / 11.4971|143.7915 / 176.3863|
|Sparse4D v3 R50 / PyTorch + custom CUDA|scene0061 (39)|0.5846|0.5453|24.7347 / 25.7167|219.3372 / 223.9781|
|Sparse4D v3 R50 / PyTorch + custom CUDA|mini_val (81)|0.4616|0.4331|24.4639 / 25.6609|217.0209 / 224.7732|
|GaussianWorld streaming / PyTorch + custom CUDA|scene0061 (39)|—|—|217.0232 / 226.6961|409.3227 / 422.1106|

模型边界并不完全相同：BEVFusion PyTorch 含上游结果回传；TensorRT 含输入搬运、后端和框回传，几何更新另计；Sparse4D 含解码/回传；GaussianWorld 的搬运单独计时。各模型顺序运行，不能把这些数字说成全部模型同时常驻的实时车端吞吐。显存和逐帧原始时延见 [metrics.csv](metrics.csv) 及各模块报告。

## 阶段证据

- **preflight：passed** — [inputs/summary.json](inputs/summary.json)
- **bevfusion_reference：passed** — [bevfusion_reference/report.md](bevfusion_reference/report.md)
- **bevfusion_tensorrt：partial** — [bevfusion_tensorrt/report.md](bevfusion_tensorrt/report.md)
- **bev_pooling：passed** — [bev_pooling/results.json](bev_pooling/results.json)
- **sparse4d：passed** — [sparse4d/report.md](sparse4d/report.md)
- **gaussianworld：partial** — [gaussianworld/report.md](gaussianworld/report.md)
- **visualize：passed** — [visualize/summary.json](visualize/summary.json)
- **summarize：passed** — [report.md](report.md)

## 已识别问题

- 默认 Myelin 检测头未满足固定精度预算，原始失败结果保留在 `bevfusion_tensorrt/attempts/myelin_head/`；LayerNorm 插件版本单独验证，没有放宽 0.005 的 mAP/NDS 下降预算。
- NVIDIA 几何区间最后一个投影的 exclusive end 边界已修复，生产 CUDA kernel 的三项测试验证修复前失败、修复后通过；真实首帧有效投影恢复为 1,835,478 个。
- BEV pooling 候选与上游保持相同累计顺序；是否获得提速以 30 轮配对结果及全模型对照为准，当前默认保持上游实现。
- GaussianWorld 无占用标签，未计算 IoU/mIoU；移除标签分支用同一组真实 anchors 隔离验证。
- Sparse4D 原生 ID 的官方 mini_val IDS 与旧 CenterPoint 单场景诊断口径不同，不直接横向比较。

## 打开与复现

本机新增结果页：[http://127.0.0.1:9094/](http://127.0.0.1:9094/)。旧回放保持原入口。视频 2 fps 是播放速度。

```bash
cd perception_lab
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage summarize
envs/rerun/bin/python tools/deployment_visualize.py --serve
```

中间张量、框差异案例与源权重对应见 [TensorRT 报告](bevfusion_tensorrt/report.md)、[几何差异](bevfusion_tensorrt/geometry_diagnosis.md)、[逐阶段计时](bevfusion_tensorrt/native_profile.md)。17 项相关单元/回归测试和 1 项真实 GaussianWorld 几何检查通过，另有 CUDA 边界与内存检查证据。

完整命令见 [run_commands.md](run_commands.md)。代码入口 [run_4090_deployment.py](../../tools/run_4090_deployment.py)，配置 [deployment_4090.yaml](../../configs/deployment_4090.yaml)。执行缓存会校验配置、权重、输入、源码和产物哈希；进程运行成功不等于精度验收成功。
