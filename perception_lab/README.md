# Perception lab — mini 教学流程

当前目标已按 BOSS 的最新要求调整为 **mini 教学闭环**：检测、分割、深度、3D 检测、跟踪、ONNX 导出与回放，其余模型作为扩展。入口：[教学页面](reports/mini_learning.html) · [学习与复现说明](LEARNING_GUIDE.md)。全量数据不再是当前阻塞项。

以下为原始执行记录；**原全量里程碑 A/B/C 均未达到，不作为当前学习目标的验收条件。** 原始需求保存在instruction/；未提供的execution_plan.yaml已按两份文档重建，其JSON语法是YAML1.2的有效子集。

已获得：COCO2017完整5000图YOLOv8s官方评估；nuScenes mini真实六相机几何；YOLO/SegFormer/Metric Depth单帧六相机推理；PointPillars/CenterPoint首帧及带历史sweep帧的原生GPU推理；YOLO单图PyTorch GPU→ORT GPU数值核对；8个COCO漏检诊断案例。最新连续场景和跟踪状态以results/status.json为准。

各模块的官方 Demo、在线 / 本地可视化入口与接入状态见 [项目首页可视化索引](../README.md#各模块可视化入口)。

统一查看多传感器与模型输出的工具选型见 [Rerun、Foxglove 等工程工具说明](../README.md#多传感器数据查看与企业工程工具)。目前已接入 Rerun，支持六相机、点云、GT、检测、跟踪及 MapTR 矢量地图预测的同步交互查看，详见 [本地启动说明](RERUN.md)；HTML + 视频保留。

## 查看结果

- reports/index.html：本地打开的总览。
- reports/final_report.md、results/metrics.csv：实测结果，scope区分完整benchmark与mini诊断。
- reports/missing_assets.md：许可数据和资源需求。
- reports/compatibility.md：依赖、数值策略、已保存补丁及未完成集成。
- TASK_STATE.md：最近运行与下一步。

## 已实际使用的命令

从 `/home/minglei/Desktop/CAR/perception_lab` 运行。编排器使用系统python3，模型由各隔离环境子进程执行。

```bash
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --phase preflight
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M00 --stage smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M01 --stage evaluation
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M01 --stage smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M02 --stage smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M03 --stage smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M04 --stage history_smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M05 --stage history_smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M03 --stage sparse_depth_diagnostic
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M12 --stage onnx_diagnostic_M01
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M14 --stage diagnostic_cases
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M05 --stage mini_scene
```

`--resume`仅在输入/配置/代码/权重哈希与产物完整性均匹配时跳过。每次新运行有独立run_id，失败日志不覆盖。当前缓存策略偏保守，共享工具变化也会触发重跑。不要同时启动多个编排进程。

`--phase baseline/evaluate/deploy/experiment/report/all` 是实际CLI选项；尚无可执行实现或缺少资产的阶段会显示blocked并返回非零，绝不将这些入口解释为全流程已经实现。未配置的legacy模型适配、TensorRT、训练以及正式val联合回放仍需继续实现和验证。

```bash
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --phase report
python3 tools/verify_artifacts.py --plan configs/execution_plan.yaml --milestone C
```

当前验收应返回非零；这是预期的诚实验收结果。示例图、mini、论文参考值和缺帧输出不能充当正式val里程碑证据。

## 环境

envs/vision与envs/mmdet3d通过venv隔离新增包，复用已有`/home/minglei/anaconda3/envs/autolabel`中的torch2.1.2+cu118。没有修改基础环境；这些venv不应被当作可直接搬运的独立环境。最终包清单保存在envs/*_freeze.txt；上游commit、权重SHA256及兼容补丁分别在configs/locked/和checkpoints/manifest.json。

单帧CUDA检查是在现有用户GPU任务并存时执行的，不是独占性能benchmark。未终止现有进程。正式计时、TensorRT构建和训练仍需稳定可用的GPU资源；正式val和训练还需missing_assets.md所列数据。

本次也已通过：

```bash
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M11 --stage mini_tracking_M05
```

两段mini视频的39帧、2fps和首尾解码已验证；这两段视频不是全部任务的正式val联合回放。10个针对性测试见logs/tests_final.log；A/B/C验收失败记录在results/verification_*.json，缺失项未被伪装为通过。

车道线功能已完成 scene-0061 的 39 帧推理，详见 [MapTR 教学与复现](MAPTR.md)。

已补齐 LiDARSeg mini 标签，当前场景 39 帧点云语义分割与真值对照已接入 Rerun，见 [LIDARSEG.md](LIDARSEG.md)。

## CenterPoint 跟踪的 Rerun 显示

已完成当前场景 39 帧的多视图跟踪展示：BEV 显示旋转框、`#ID` 和最近最多 20 个显示位置的轨迹；六路相机显示三维框投影与 `#ID`；同一 ID 在 BEV、相机和三维视图中使用一致颜色。LiDARSeg 继续负责点云语义分割，跟踪独立使用 CenterPoint + PubTracker。

```bash
# 从 perception_lab 目录运行；使用已生成的检测和跟踪结果
# 重新导出为可选操作，已有最新记录时只需启动并刷新页面
envs/rerun/bin/python tools/rerun_mini.py --score 0.25
envs/rerun/bin/python tools/start_rerun.py
```

[打开本机查看器](http://127.0.0.1:9090/?url=http%3A%2F%2F127.0.0.1%3A9091%2Fmini_scene.rrd&renderer=webgl)。启动后播放或拖动时间轴；密集标签可放大视图查看。相机投影未做动态目标曝光时间补偿和遮挡判断；ID 可能因漏检或关联错误改变。

代码：[跟踪与坐标转换](tools/track_mini.py)、[投影与配色](tools/tracking_overlay.py)、[Rerun 导出](tools/rerun_mini.py)。图层路径和操作见 [RERUN.md](RERUN.md)。本次更新通过 28 项测试、39 帧记录格式校验和本机 Chrome 播放检查，未进行全量跟踪精度评估。

## 检测与跟踪评估

已完成 scene-0061 的 39 帧固定阈值诊断（score ≥0.25，同类别中心距离 <2 m）。检测 TP/FP/FN 为 2235/1020/75，跟踪为 1177/579/39；连续帧 ID 切换 46 次，间隔后换 ID 38 次。计数为逐帧目标次数，检测和跟踪的类别范围不同；未运行官方全量 mAP/NDS/AMOTA 评估。

```bash
envs/mmdet3d/bin/python tools/evaluate_mini.py --score 0.25 --distance 2.0
```

[完整报告与 10 张案例图](reports/mini_evaluation/report.md)、[本地 HTML 画廊](reports/mini_evaluation/index.html)、[全部案例](reports/mini_evaluation/cases.csv)、[逐帧统计](reports/mini_evaluation/frames.csv)、[核验结果](reports/mini_evaluation/verification.json)。案例的 frame 从 0 开始，与 Rerun 对应；完整过滤、匹配及身份事件定义以报告为准。脚本见 [evaluate_mini.py](tools/evaluate_mini.py)，匹配逻辑见 [evaluation_utils.py](tools/evaluation_utils.py)。
