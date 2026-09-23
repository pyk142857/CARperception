# mini 感知流程学习

BOSS 已确认：仅用 nuScenes mini，先完成现有模型的教学闭环，其余模型作为扩展。不需要下载 trainval、Cityscapes 或占用标签；不要求正式榜单精度。新范围记录在 configs/execution_plan.yaml 的 active_scope 中，旧 jobs 与 A/B/C 验收保留作历史参考。

先打开 [教学回放](reports/mini_learning.html)，再按下表阅读代码和原始输出。

| 环节 | 要理解的问题 | 当前演示范围 |
|---|---|---|
| 数据与标定 M00 | 各传感器时间不同，点为何能投影到相机？ | 首时刻六路相机；GT 仅用于标注展示 |
| YOLO M01 | 图像经过何种预处理，框和置信度如何输出？ | 首时刻六路相机 |
| SegFormer M02 | 像素类别编号如何变成彩色分割图？ | 首时刻六路相机 |
| Metric Depth M03 | 深度数组与彩色图有什么区别？ | 首时刻六路相机，深度单位米 |
| PointPillars M04 | 历史点云如何对齐到当前时刻？ | 单帧与带历史 sweeps 帧 |
| CenterPoint M05 | 点云如何变成带类别和速度的三维框？ | scene-0061 连续 39 帧 |
| 跟踪 M11 | 同一目标跨帧如何保留 ID？ | 消费 M05 真实检测，39 帧 |
| 导出 M12 | ONNX 是否保持模型输出？ | 同一 mini 前视图，PyTorch GPU 对比 ORT GPU |
| 回放 | 相机图、检测和轨迹如何按帧对应？ | 三栏同步视频，39 帧，2 fps |

图像的检测、分割、深度是并列分支。PointPillars 与 CenterPoint 也是并列模型；本次跟踪只接 CenterPoint。雷达用于数据读取和投影理解，未完成雷达融合模型。视频中的前视图是原始图，不包含连续 YOLO 预测。播放速度不代表模型实时性能。

## 验证并重建教学页面

从 perception_lab 目录执行：

```bash
envs/vision/bin/python tools/mini_learning.py
```

该命令检查已保存输出的 SHA256，核对 39 个传感器 / 检测 / 跟踪 sample token，验证 mini 导出校验记录，生成同步视频并逐帧解码，再生成页面和 results/mini_learning.json。它复用现有真实推理输出，不会下载数据或重新跑全部模型。

## 逐环节重新推理

```bash
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M00 --stage smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M01 --stage smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M02 --stage smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M03 --stage smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M04 --stage history_smoke
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M05 --stage mini_scene
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M12 --stage mini_onnx_M01
```

每次输出在新的 runs 目录，不覆盖旧记录。跟踪必须接上一次 M05 输出；执行以下命令把跟踪输入更新为当前 M05 的文件，然后运行跟踪与教学检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p=Path('configs/execution_plan.yaml')
plan=json.loads(p.read_text())
status=json.loads(Path('results/status.json').read_text())
record=status['modules']['M05']['mini_scene']
assert record['status']=='passed'
detections=str(Path(record['log']).parent/'predictions.json')
job=plan['modules']['M11']['jobs']['mini_tracking_M05']
job['command'][job['command'].index('--detections')+1]=detections
job['inputs'][0]=detections
p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
PY
python3 tools/run_pipeline.py --plan configs/execution_plan.yaml --module M11 --stage mini_tracking_M05
envs/vision/bin/python tools/mini_learning.py
```

不要用旧的 `--phase all` 作为教学入口，它仍包含完整基准和扩展任务。M06–M10、TensorRT 与训练不属于本次教学必需项；没有声称它们已经完成。既有 COCO 评测保留作为附加材料。
