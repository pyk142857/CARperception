# Instruction 20260928_01：用 nuScenes 官方代码评估 M05 CenterPoint 检测

## 任务与数据范围

Codex：在 CARperception 所在工作站实际实现、运行并报告本指令。先阅读根目录 `AGENTS.md`、`perception_lab/TASK_STATE.md`、`perception_lab/reports/mini_evaluation/report.md`、`perception_lab/tools/lidar_smoke.py`、`lidar_scene.py`、`evaluate_mini.py` 及已有指令。只处理 **M05 CenterPoint 的 3D 检测评估**；保留已有固定阈值失败案例和 M11 跟踪实验的原口径。不要只提交公式、接口或计划。

当前已有 `scene-0061` 的 39 帧实测预测，`results/status.json` 的 `M05.mini_scene` 指向实际 run；先检查本机原始 `predictions.json`、`data/nuscenes/v1.0-mini` 和已安装 `nuscenes-devkit`，优先复用预测，避免重新运行 GPU。现有全量 `v1.0-trainval` val 数据和 M05 全 val 预测尚未证实可用。

两级成果：

1. **立即完成**：使用官方 `nuscenes.eval.detection.evaluate.DetectionEval` 和 `detection_cvpr_2019` 配置，对完整 `scene-0061` 运行自定义 mini 场景 split。报告写作“官方评估代码计算的 mini 单场景诊断”，不能写成 nuScenes 官方 val 榜单成绩。
2. **数据具备时继续**：若本机存在完整 `v1.0-trainval` val 数据以及 M05 对全部 val sample 的真实预测，使用同一导出工具直接运行官方 `--eval_set val`。缺任何一项就记录 `blocked_full_val` 和确切缺失项，不补造预测、不把 mini 结果改名为全 val。完成 mini 评估不以全量数据下载或训练为前置条件。

## 1. 检查输入、版本与覆盖

- 从 `results/status.json` 找到 `M05.mini_scene` 的原始实测文件，通常为 `runs/20260923T043110_d6ae8066/M05/mini_scene/predictions.json`，以实际文件为准。确认 `source=measured`、checkpoint/config 哈希、39 个唯一 `sample_token`；与 nuScenes 元数据中 `scene-0061` 的**全部** sample 集合严格相等。原模型真实输出为空的帧仍须有 `[]`；原始文件缺帧时先定位缺失并重跑缺帧，禁止用 GT 或作者下载的预测补帧。
- 核对当前评估环境中的 `nuscenes-devkit` 版本及 `DetectionEval`、`get_samples_of_custom_split`、`get_scenes_of_custom_split` 是否可用。若现有版本不支持自定义场景 split，在**独立评估环境**使用可核对的官方 devkit 版本并记录版本/commit；不要破坏现有 mmdet3d 推理环境，不修改官方的匹配、过滤、`accumulate` 或 `calc_ap`。
- 记录数据版本、模型 config/权重来源与 SHA256、预测来源和分数范围。检查原模型已经实施的分数过滤和 NMS；导出环节不要再叠加固定 `score >= 0.25` 截断。

## 2. 仅实现预测格式与坐标适配

新建可复用的导出工具，例如 `tools/export_nuscenes_detection.py`：读取 M05 的 `predictions.json`，输出官方 detection result JSON，顶层为 `meta`、`results`。`results` 必须包含目标 split **每个** sample token 的键，其值为该帧的预测列表（允许空列表）。`meta` 如实标明 M05 使用 LiDAR；不能把数据集中有相机或毫米波雷达误记成模型输入。

每个检测框包含：`sample_token`、`translation`（global xyz，米）、`size`（w,l,h，米）、`rotation`（global 四元数 w,x,y,z）、`velocity`（global xy，米/秒）、`detection_name`（官方 10 类）、`detection_score`（模型输出分数）、`attribute_name`。现有 `boxes3d` 中的 `center_xyz`、`rotation_wxyz`、`velocity_xy` 属于 ego 系，必须用**该 sample 对应的 ego_pose**变换到 global；`size_wlh` 顺序不变。注意原生 LiDAR bottom-centre 与标准化 gravity-centre 的区别，不重复移动框中心。M05 若缺真实速度，直接报错并定位来源；属性仅在模型确有预测时填写，否则写空字符串并在报告说明 mAAE/NDS 的边界，绝不从 GT 注入属性。

验证：所有数值有限、正尺寸、单位四元数、类别合法、分数在 `[0,1]`、每帧最多 500 框。若原模型单帧输出超过 500，只能按分数稳定截取前 500 并记录被移除数量。挑选至少一个真实框检查 ego→global→ego 的中心/速度往返误差及姿态组合；GT 只能用于离线几何核验，不能进入预测生成。记录输入/输出 SHA256 与导出命令。

## 3. 调用官方评估器

若已装 devkit 支持自定义 split，在本地 `data/nuscenes/v1.0-mini/splits.json` **合并** `{"mini_scene_0061":["scene-0061"]}`，不要覆盖原有定义；此数据目录遵守现有忽略规则。用 devkit 读取该 split 的完整 sample 集合，确保与导出的键集合一一对应。然后用同一评估环境直接调用官方 CLI；下列 run ID/环境路径仅作已知入口，实际报告必须记录真实运行命令：

```bash
cd /home/minglei/Desktop/CAR/perception_lab
envs/mmdet3d/bin/python -m nuscenes.eval.detection.evaluate \
  runs/20260923T043110_d6ae8066/M05/mini_scene/nuscenes_detection.json \
  --eval_set mini_scene_0061 --version v1.0-mini \
  --dataroot data/nuscenes \
  --output_dir reports/official_detection_mini/metrics \
  --plot_examples 5 --render_curves 1
```

若需独立环境，仅替换 Python 路径。保留官方 `detection_cvpr_2019`：官方各类范围、地面中心距离 `0.5/1/2/4 m`、`min_recall=0.1`、`min_precision=0.1`、最多 500 框/帧。由官方代码完成按分数排序的一对一匹配、PR 曲线、40 个类别×门限 AP、mAP、五项 TP 误差及 NDS。不要以自写 AP、匈牙利匹配或 `evaluate_mini.py` 的单点 P/R 取代官方输出。

如果本机有完整官方 val 数据及相应模型预测，再单独运行 `--eval_set val --version v1.0-trainval`，输出到 `reports/official_detection_val/`；执行前核实版本、split 与 sample token 全覆盖。mini 结果与 val 结果各自保存，不混用。

## 4. 验收、报告与提交

- 提交导出工具和必要的最少复现说明；原有 `reports/mini_evaluation/` 与 `tools/evaluate_mini.py` 保持原有诊断口径。使用官方代码对 mini 全场景真实运行成功，重新读取输出的 `metrics_summary.json`、`metrics_details.json` 与 PR 曲线，校验 39 帧、JSON 字段和坐标转换无误，方可标记此任务通过。
- 在 `reports/official_detection_mini/` 留存报告、官方 JSON 结果、每类四距离 AP 摘要、至少一张官方曲线或示例、日志、最终执行命令、模型和数据版本、输入/输出 hash。大文件按仓库已有忽略规则保存在本机，同时在报告留下位置和生成方式。
- 报告分清：① 原 `score≥0.25/2 m` mini 单点 P/R；② 本次官方代码 mini 单场景 AP/mAP/NDS；③ 完整官方 val 指标（实测或 `blocked_full_val`）。说明 mini/预训练可能重叠以及空属性对属性误差的影响。原 M05 的 full val evaluation 缺数据时仍为 pending/blocked；另记录 mini 官方代码评估已通过。
- 更新 README 的报告入口和 `TASK_STATE.md`，按根目录 `AGENTS.md` 提交并推送代码和报告，核对远端文件。执行报告写明提交号、真实命令、实测数值、未完成项及继续所需数据。不宣称没有运行的评估已完成。

官方资料：[任务与格式](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md) · [评估入口](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py) · [自定义 split](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py) · [匹配与 AP](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py) · [默认配置](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/configs/detection_cvpr_2019.json)。
