# MambaFusion：原生 39/81 帧与官方检测评价

作者完整权重已恢复，严格加载全部 **1,403 个状态条目**，missing/unexpected 均为0。真实六相机＋10 sweeps 完成120帧；使用原始 global results/meta 直接运行官方 DetectionEval，没有重复坐标转换、显示阈值裁剪或GT属性填充。

| 范围 | 帧 | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene0061 | 39 | 0.727975 | 0.673878 | 0.299275 | 0.322943 | 0.334202 | 0.344613 | 0.600066 |
| mini_val | 81 | 0.588979 | 0.595775 | 0.396695 | 0.430137 | 0.445999 | 0.424399 | 0.289916 |

这是 nuScenes mini 教学诊断，使用 detection_cvpr_2019 官方评价协议，不是官方全量验证成绩；预训练可能覆盖样本。39帧采用自定义scene-0061 split，81帧采用官方 mini_val。各类AP、config、预测/指标/GT/split哈希见各split/official_eval/。原始预测完整保存在 scene0061_predictions.json 和 mini_val_predictions.json。

## 来源与实际兼容改动

源码 `AutoLab-SAI-SJTU/MambaFusion@d85c2feeb19207b20e42d50421afb11c095f3efa`；作者权重391,545,251字节，SHA256 `d2454354b762a5a4b0b85af8b23c4b1f63da9357af7160aa313bc689a8d11ea6`。该hash来自源端完整下载与本机复核，不称作者预公布校验值。

沿用Torch2.1.2/cu118隔离环境、原生CUDA扩展和Hilbert模板。作者config固定六BGR相机256×704、点特征XYZ/intensity/time、范围[-54,-54,-5,54,54,3]、voxel[.3,.3,.25]、200proposals；无训练、TTA或模型替换。两个NumPy兼容改动只涉及废弃np.int别名与异构图像处理元数据的显式object数组；[补丁与复现](../../perception_weights_recovery_20261013/mambafusion_numpy_compat.md)。历史550文件来源证明保留，当前差异另列。网络严格加载未删键、reshape或随机补参。

## 重复噪声与验收边界

六个首/中/末帧各重复两次，数值输入哈希一致，实际相机/点云/高度编码/融合/scan调用与有限值检查通过。原生候选数、顺序和部分预测会变化；临时设置的按顺序0.1/0.01一致性门**未通过**，未改写为通过。

[保存的重复诊断](repeat_noise_review.json) · [源码可能来源与限制](repeat_noise_source_review.md)。本轮显式允许保留已测噪声的原生mini诊断评价，所有原始预测继续参与官方评价；不证明确定性，不批准优化数值等价。正式输出对应一次完整推理，绑定[执行回执](native_receipt.json)。执行时适配器另存 executed_native_adapter.py；之后只增强了复现校验，不冒充重新推理。

## 耗时与失败案例

- scene0061：带诊断 forward 均值 697.17 ms，端到端 990.85 ms，Torch 峰值分配 8.976 GiB。
- mini_val：带诊断 forward 均值 518.49 ms，端到端 807.96 ms，Torch 峰值分配 4.193 GiB。

以上包含有限值Hook同步、首次加载和诊断开销；另观察到其他项目GPU任务，未采集逐帧外部占用。因此不是独占性能基准，不能据此计算优化加速比。详见 timing_limitations.json。

独立失败诊断采用官方GT/距离/有效点过滤、score≥0.25、同类XY中心距离<2m贪心匹配：scene0061 TP2155/FP103/FN155，mini_val TP3135/FP414/FN258。FN无预测置信度。案例、距离/类别/分数和前三张图见 [39帧](scene0061/badcases/cases.json)、[81帧](mini_val/badcases/cases.json)；这不是AP本身。

## 复现（项目根目录）

```bash
python3 perception_lab/tools/mambafusion_numpy_compat_2025.py
perception_lab/envs/mambafusion_2025/bin/python perception_lab/tools/mambafusion_2025.py --phase gate --config perception_lab/configs/perception_2025.yaml
# 保留并检查真实六帧重复证据；当前发布的review绑定repeat_diagnostic_baseline。
perception_lab/envs/mambafusion_2025/bin/python perception_lab/tools/mambafusion_2025.py --phase run --allow-measured-repeat-noise --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/nuscenes_eval/bin/python perception_lab/tools/mambafusion_2025.py --phase evaluate --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/nuscenes_eval/bin/python perception_lab/tools/mambafusion_badcases_2025.py
```

旧下载失败记录和原报告保存在恢复报告的prior_snapshot，未删除历史失败。新Mamba→GRAE与同步回放验收独立记录，见上级报告。
