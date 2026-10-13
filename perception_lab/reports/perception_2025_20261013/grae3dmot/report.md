# GRAE-3DMOT：nuScenes mini 新推理结果

真实官方权重已严格加载，已在冻结的历史检测输入上重新执行 39 + 81 帧 GRAE 推理和官方 TrackingEval。这里复用的是检测输入文件，跟踪输出与指标均为本次生成；不代表全量 nuScenes val，也不代表 MambaFusion 检测链路已经完成。

| 输入 | 范围 | AMOTA | AMOTP | IDS | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| frozen_reference | scene0061 / 39 | 0.862569 | 0.404881 | 5 | 67 | 104 |
| frozen_reference | mini_val / 81 | 0.822964 | 0.435186 | 13 | 299 | 400 |

## 来源、兼容性与推理边界

作者源码固定到 `63def8bde5e199a4e77fdf4fab76a4b3511fe132`，通过官方 codeload 完整快照获取（Git 传输失败，不宣称本地是完整 Git clone）。官方 `model.pth` 为 7,289,485 字节，SHA256 `9da45f9cfce1b4949bcd92d61193b88e09fea59c37f7b0a963249aad83728395`。配置为作者 `config/val.json`，128 channels、3 layers；162 个 state tensors、596,906 个唯一参数严格加载，无 missing/unexpected。

首次直接 strict load 因发布权重 `spatial_mfl`/`temporal_mfl` 与固定源码 `spatial_dffl`/`temporal_dffl` 的属性命名不一致而失败，原日志保存。经逐模块架构核对，仅做这两个精确前缀映射，共 16 个键，不改张量内容或次序；加载后全部张量与原始权重 `torch.equal`，逐键 shape/hash 见 `checkpoint_key_mapping.json`。没有 random weights、`strict=False`、训练或删键。

隔离环境复用现有 Torch 1.13.1+cu117 只读包路径，新增 LAP/fvcore 相关小依赖，未修改旧环境。SimpleTrack 仅导入原 NMS（阈值 0.1），没有运行旧跟踪器。推理使用七个跟踪类别及作者分数阈值 0.1。输入为 global xyz、wlh、wxyz quaternion、xy velocity，所有有限值与形状检查通过。新建 mini reader 只允许预测字段，GT ID 和 instance_inds 固定为 -1；真实时间戳由固定 packet 读取，120 帧与 LIDAR_TOP timestamp 完全一致；真实间隔约 0.399–0.600 s，由作者 time_lag 路径使用。

作者原生关联逻辑与状态更新保持。空间/时间投影、距离融合和 affinity 模块实际 hook 调用且输出有限；spatial 137 次、temporal 125 次包含独立验证帧，不将其误报为数据集帧数。首/中/末帧、空帧、GT sentinel 替换不影响输出、场景重置与固定前缀 replay 恢复检查均通过。恢复方式是重放冻结前缀，不是声称存在新的隐藏状态持久化 API。

## 性能与优化

30 组交错 AB/BA，每组连续 3 帧、双方各 3 次 warmup，对比原生输出循环与批量最终 D2H 导出。完整跟踪输出 30/30 完全相等。原生均值 34.7894 ms/3帧，批量导出 33.9916 ms/3帧，平均节省 0.7978 ms；但 AB 顺序均值为 -0.0407 ms，BA 为 +1.6362 ms，具有明显顺序依赖，因此没有稳定加速结论，保留原生方案。原 30 组未记录双方独立显存峰值，不将单独 native profile 峰值冒称 A/B 显存对比。该计时从预处理后的 CPU detection 到可消费跟踪输出，排除 JSON 与 NMS，不能冒充端到端收益。

单独 CPU 预处理 120 帧耗时约 25.11 s，作者 NMS 是明显耗时项：39 帧均值 139.56 ms/帧，81 帧均值 240.84 ms/帧。不能把导出的小幅波动写成全链路改善。原始完整推理性能文件来自含 finite hooks 的验证运行；分段 CUDA events、冷模型加载与 RAM profile 已单独实测，见 `segmented_profile.json`。连续 3 帧：冷加载 1228.76 ms；输入 JSON 读取 42.17 ms；allowlist+NMS 345.03 ms；CPU instance 0.27 ms；H2D+初始状态 CUDA events 0.783 ms；3 次 forward 合计 4.551 ms；关联+导出 wall 26.65 ms，其中最终 D2H/记录构建 7.20 ms；原始检测文件到可消费 tracks 415.16 ms。GPU allocated 峰值 11,777,024 bytes，reserved 33,554,432 bytes；进程生命周期 RAM high-water 2,015,973,376 bytes。计时器编译在 wall 起点前完成，插桩与原生输出完全一致。forward 与导出是关联 wall 内的子区间，不能相加；输入已是 global 坐标，无额外坐标变换操作；可视化不在此 tracker 推理边界。初次把计时器编译计入 wall 的测量以 instrumentation_overhead 文件独立保留并明确 superseded。

## 独立教学诊断

GT 只在推理完成后进入官方评价和教学审计。教学审计采用 score≥0.25、同类 XY<2m，先保留有效历史匹配再 Hungarian；不做官方轨迹插值，所以 IDSW 不等同于官方 IDS。

| 范围 | TP | FP | FN | 即时 IDSW | gap 后换 ID | gap 后恢复同 ID |
|---|---:|---:|---:|---:|---:|---:|
| 39 帧 | 1030 | 20 | 186 | 5 | 15 | 79 |
| 81 帧 | 2821 | 205 | 537 | 12 | 12 | 138 |

每个范围有 3 张真实输出/GT BEV 对照图及事件完整列表（对应 `frozen_reference/<split>/teaching_audit.json` 和 `teaching_case_*.png`）。这里的 gap 是阈值匹配中断，不能仅凭此证明物理遮挡、检测缺失或某个模块的因果改善。代表例覆盖即时 ID 切换、长 gap 后换 ID、长 gap 后同 ID 恢复。

官方指标、每类详情、预测结果、state 检查、输入 SHA、参数映射证明、30 组原始计时和复现命令均保留。MambaFusion 输入链路尚待原生检测结果，因此整体状态为 partial；不会把 frozen_reference 成绩代称为 MambaFusion+GRAE 成绩。
