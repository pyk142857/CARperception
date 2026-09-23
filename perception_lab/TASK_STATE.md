# TASK_STATE — mini 教学闭环已验证

2026-09-23，BOSS 明确改为理解流程：仅用 nuScenes mini，现有模型优先，其余模型作为扩展。

## 当前完成范围

- M00 数据 / 标定投影；M01 YOLO、M02 SegFormer、M03 Metric Depth：首时刻六路相机真实结果。
- M04 PointPillars：单帧与历史 sweeps；M05 CenterPoint：scene-0061 全部 39 帧。
- M11 作者跟踪器：消费本次 M05 检测，39 帧。
- M12：新增 mini 前视图 PyTorch GPU → ONNX Runtime GPU 数值核对，通过。
- 新增三栏教学回放：原始前视图、CenterPoint 三维框、跟踪 ID / 轨迹，39 帧，2 fps。
- 教学检查核对八个阶段既有产物的 SHA256、逐帧 token、视频完整解码与导出记录。既有推理输出复用，不宣称本轮全部重新推理。

## 入口

- reports/mini_learning.html：主学习入口。
- LEARNING_GUIDE.md：输入 / 输出说明、学习顺序、逐模块复现命令。
- results/mini_learning.json：教学验收证据与 run ID。
- outputs/replay/mini_learning.mp4：同步回放。
- configs/execution_plan.yaml 的 active_scope：当前范围；旧 jobs 仍保留。

## 范围边界

原全量 A/B/C 未达到，但不再是当前目标。全量 nuScenes / Cityscapes / 占用标签不再阻塞本教学闭环。M06–M10、TensorRT、训练微调为扩展且尚未完成；已有 COCO 评测为附加历史结果。未宣称多传感器模型融合、实时性能或全量泛化成绩。

模型运行命令见 LEARNING_GUIDE.md；不要直接运行旧 --phase all，它包含全量评测和扩展。当前无新后台训练或下载任务。
