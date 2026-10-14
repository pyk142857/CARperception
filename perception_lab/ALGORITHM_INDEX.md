# 算法实现与优化入口

本项目以 nuScenes mini 教学与可复现实验为目标。下表区分已验证的优化、可选后端和未采用候选；收益只适用于各自报告中的输入与计时范围。代码、配置、指标与失败证据均随仓库发布。

## 已验证的优化

| 方案 | 算法代码 | 运行与配置 | 验证证据 |
|---|---|---|---|
| TileLang FP64 旋转框三维 IoU / GPU NMS | [kernel 与运行适配](tools/tracking_gpu_kernel_20261013.py)、[原几何语义与输入打包](tools/tracking_gpu_geometry_20261013.py) | [GRAE GPU 入口](tools/tracking_gpu_grae_20261013.py)、[配置](configs/tracking_gpu_20261013.json)、[阶段 runner](tools/run_tracking_gpu_20261013.py) | [完整报告](reports/tracking_gpu_20261013/report.md)、[120 帧正确性](reports/tracking_gpu_20261013/kernel_correctness.json)、[三层计时](reports/tracking_gpu_20261013/paired_benchmark.json) |
| CenterPoint 两阶段贪心关联 S4_greedy_low | [参考跟踪器](tools/experimental_tracker.py) | [实验配置](reports/tracking_optimization/configs.json) | [13 组策略对照](reports/tracking_optimization/report.md) |
| FastCandidateTracker CPU 关联优化 | [优化实现](tools/fast_candidate_tracker.py) | [配对基准](tools/benchmark_tracker_operators.py) | [输出一致性与性能](reports/tracking_operator_optimization/report.md) |

TileLang 替换 GRAE 的几何/NMS 预处理，原网络、权重、关联与导出保留；120 帧轨迹和官方指标一致，完整跟踪链路平均耗时降低87.09%，不含 BEVFusion 检测前向。当前容量256框、真实最大199框，默认阈值0.1/1.0/0.3，使用FP64几何。CPU两阶段策略与算子优化对应另一条 CenterPoint 链路，不能将两条链路的提速倍数相乘。

## 可选后端与未采用候选

| 方案 | 实现 | 当前决定及证据 |
|---|---|---|
| DEIMv2 模型与后端部署 | [适配入口](tools/deimv2_2025.py) | 原生模型优先；TensorRT FP16为有一致性限制的可选后端。[报告](reports/perception_2025_20261013/deimv2/report.md) |
| BEV pooling 索引预计算 | [CUDA kernel](tools/bev_pooling_preindex_20261013.cu)、[局部基准](tools/bev_pooling_preindex_20261013.py)、[隔离整网接入](tools/bev_pooling_preindex_isolate_20261013.py)、[整网汇总](tools/bev_pooling_preindex_summary_20261013.py) | 局部收益约5%–10%，整网稳定收益未通过，保留上游实现。[失败分析与评价](reports/tracking_gpu_20261013/bev_pooling/report.md) |
| OptiPMB 原生跟踪 | [真实点云输入与原生适配](tools/optipmb_tracking_20261013.py) | 81帧精度替换门槛未通过，保留GRAE。[对照报告](reports/tracking_gpu_20261013/report.md) |

模型选型、其他候选及适用范围见[保留方案](RETAINED_ALGORITHMS.md)。实验代码仍保留在仓库；存在代码不表示该候选已成为默认后端。

## 复现与核验

先按[隔离环境说明](reports/tracking_gpu_20261013/environment/README.md)准备指定版本的依赖、权重和数据，再使用[实际运行命令](reports/tracking_gpu_20261013/run_commands.md)。GPU运行依赖本地编译库与冻结输入；仓库不包含原始数据集、权重、虚拟环境或大型二进制产物。

从仓库根目录只读核验已发布的实验记录：

```bash
python perception_lab/tools/tracking_gpu_acceptance_20261013.py --scope all
```

该命令校验已保存的证据，不会重新运行GPU实验。重跑推理、编译或基准可能更新对应实验输出，应先按复现说明确认环境与输出目录。
