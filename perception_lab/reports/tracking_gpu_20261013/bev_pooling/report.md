# BEV pooling 索引预计算：实测完成，保留原实现

完成局部及隔离整网实验，**不采用候选**。局部输出逐位一致，耗时下降约5%–10%；9个整网代表帧均未达到24/30更快的门槛，局部收益不足以支持替换。

## 局部30轮AB／BA

| Capture | 原实现 ms | 候选 ms | 更快轮次 |
|---|---:|---:|---:|
| frame_0 | 0.225810 | 0.202555 | 30/30 |
| frame_19 | 0.215961 | 0.199515 | 30/30 |
| frame_38 | 0.215563 | 0.204177 | 28/30 |

四组边界用例与三份真实capture均通过；保留FP16布局和原FP32 FMA顺序。候选计时包含每次forward重新计算索引、清零和pooling，不跨帧摊销。每次新增7,974,912字节索引workspace。输入来源、哈希、布局和逻辑访问估计见[input_analysis.json](input_analysis.json)；访问量估计不是硬件带宽测量。

## 整网官方检测评价

范围为nuScenes mini：scene-0061 39帧与mini_val 81帧。使用原DetectionEval/detection_cvpr_2019，配置、权重、ONNX和TensorRT plans保持相同，双方使用同一个隔离重建库；全部实际预处理输入哈希相同。新预测独立保存，冻结跟踪输入未改动。

| 范围 | mAP 原实现→候选 | NDS 原实现→候选 |
|---|---:|---:|
| scene0061 | 0.710627 → 0.710133 | 0.621758 → 0.621536 |
| mini_val | 0.576824 → 0.576630 | 0.513460 → 0.513240 |

两组mAP/NDS下降均小于0.005，整体P95几何预算均通过；逐类AP、逐框匹配及绑定评价哈希见[整网摘要](full_model_summary.json)及full_model下的comparison文件。这是mini教学诊断，不是全量验证。

## 整网30轮AB／BA

| split / token前8位 | 原实现 ms | 候选 ms | 节省 | 更快轮次 |
|---|---:|---:|---:|---:|
| scene0061 / ca9a282c | 8.1601 | 7.9504 | 2.57% | 17/30 |
| scene0061 / ed5fc18c | 8.7298 | 8.7072 | 0.26% | 16/30 |
| scene0061 / b26e7915 | 9.2694 | 9.2501 | 0.21% | 16/30 |
| mini_val / b4ff3010 | 9.0584 | 9.0961 | -0.42% | 14/30 |
| mini_val / b5989651 | 8.1585 | 7.9862 | 2.11% | 18/30 |
| mini_val / 3e8750f3 | 8.3414 | 8.1423 | 2.39% | 16/30 |
| mini_val / 5b03af7a | 8.8852 | 8.5246 | 4.06% | 18/30 |
| mini_val / 281b9226 | 9.1796 | 8.7562 | 4.61% | 18/30 |
| mini_val / 6ff9723a | 9.3864 | 9.3325 | 0.57% | 17/30 |

时间范围为预处理后的固定CPU输入→H2D→模型→检测框D2H；几何更新、预处理、加载和capture导出不在配对计时内。各序列timings.csv另外记录预处理、几何更新和端到端时间；performance.json记录显存（新增workspace包含在进程显存中）。同一输入双方分别预热10次。原始270组整网配对CSV、P50/P95和AB/BA分别均值均保留，不能把均值微小改善当成稳定提速。GPU竞争监控0.5秒（局部）／1秒（整网）及边界采样均无其他compute进程，见gpu_monitor.json和full_model_gpu_monitor.json；短于采样间隔的竞争不能完全排除。

## 一致性限制与失败证据

隔离重建库与旧库首帧输入一致，但输出不是逐字节一致，匹配后的yaw P95约0.134rad，超过0.1rad诊断预算，见full_model/smoke/rebuild_geometry.json。没有把此失败改写为通过。固定同一输入分别重复原实现和候选各5次，两种模式均观察到检测输出变化，见full_model/repeat/repeat_outputs.json和摘要；该观察证明存在重复运行波动，但未完成算子级因果定位。

两次独立整网运行中，6份capture的indices及intervals全相同；其中5份camera features/depth及pool输出不逐位相同。这些capture不能直接当作“同输入pooling误差”；严格同输入证据来自前述独立局部实验。尚未证明整网张量逐位等价，因此准确率通过也不支持采用。

最终决策：研究阶段已完成，**保持上游默认实现**，候选仅保留为可复现实验。GPU跟踪优化的精确等价与87.09%收益是独立结论。

## 复现与来源

隔离构建与命令见[isolated_full_model_plan.md](isolated_full_model_plan.md)。该方案本次已执行；构建日志isolated_build.log，实际脚本归档execution/，每次运行日志full_<mode>_<split>.log。首次误用tensorrt环境缺少torch，在启动模型前失败，日志保留；正式局部运行改用optipmb_20261013环境，整网使用bevfusion环境。轻量runner已修正。

汇总命令：`perception_lab/envs/sparse4d/bin/python perception_lab/tools/bev_pooling_preindex_summary_20261013.py`。原始capture、隔离库、环境仍在本地，未上传原始数据或权重。
