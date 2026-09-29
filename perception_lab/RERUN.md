# 本地 Rerun 查看器

本集成复用已经运行完成的 mini 结果，不重新调用 GPU 模型。查看 `scene-0061` 的 39 个关键帧：六路相机、激光点云、GT 三维框、CenterPoint 检测框、PubTracker 跟踪框和轨迹。

## 独立环境

从 `perception_lab` 目录执行，使用 Python 3.10 创建独立环境：

```bash
python3.10 -m venv envs/rerun
envs/rerun/bin/python -m pip install -r configs/rerun_requirements.txt
```

固定使用 Rerun 0.23.4，使导出端和查看器版本一致，不改动 `vision` / `mmdet3d` 推理环境。配置和 API 参考 [对应版本文档](https://ref.rerun.io/docs/python/0.23.4/common/initialization_functions/)。

## 生成记录

```bash
envs/rerun/bin/python tools/rerun_mini.py
```

需要本机已经存在 `manifests/frame_packets.json`、mini 原始数据、`results/status.json` 及其指向的 M05 / M11 完整运行产物。克隆代码不会同时获得这些大文件，请先执行教学流程。导出前验证运行产物哈希及逐帧 token 一致性。

输出位于 `outputs/rerun/mini_scene.rrd`，相邻 JSON 保存帧统计、输入与输出哈希。默认仅展示分数不低于 0.25 的预测 / 跟踪框，可用 `--score 0.1` 改变阈值重新生成。

## 查看方式

从 `perception_lab` 目录执行：

```bash
envs/rerun/bin/python tools/start_rerun.py
```

默认在本机启动 Web 查看器（9090）、记录文件 HTTP 服务（9091）和 Rerun 自带 gRPC 端口（9876），全部绑定 `127.0.0.1`。命令会打开浏览器；`--no-browser` 仅启动服务并打印地址。重复启动会复用已有服务。

[打开本地查看器](http://127.0.0.1:9090/?url=http%3A%2F%2F127.0.0.1%3A9091%2Fmini_scene.rrd&renderer=webgl)

```bash
# 停止本项目启动的两个服务进程
envs/rerun/bin/python tools/start_rerun.py --stop
# 自定义端口（端口冲突时）
envs/rerun/bin/python tools/start_rerun.py --port 9190 --data-port 9191 --grpc-port 9976
```

为避开当前机器的文件监听数量限制，启动器通过 HTTP 向查看器提供保存好的 `.rrd`，不使用文件监视或实时推理。重新导出记录后刷新浏览器即可加载；没有设置开机自启。运行 PID 与日志位于 `outputs/rerun/server.json`、`server.log`。

已在本机有图形环境的 Chrome 中验证 WebGL 渲染。自动化无头浏览器可能缺少足够的图形能力；请用本机普通 Chrome 打开。查看器代码及记录均由本机服务提供。

## 界面操作

- 在底部时间轴选择 `frame` 或 `elapsed`，拖动时间游标、暂停或播放。
- 在三维视图中拖动旋转，滚轮缩放；六路相机面板随时间更新。
- 展开左侧 Blueprint 中的三维视图，通过实体可见性开关比较 `ground_truth`、`centerpoint`、`tracks` 和 `trails`。
- GT 为绿色、检测为橙色；跟踪按 ID 固定配色，在 BEV、三维和相机视图中一致。跟踪标签默认显示，GT 与检测标签默认隐藏。
- 原始相机图目前没有连续二维检测 / 分割 / 深度图层；这些模型仅有首时刻结果，不自动补帧。

## 坐标和时间语义

共同参考系为当前 LiDAR 时刻的车体坐标：x 前、y 左、z 上，单位米。点云使用 LiDAR 外参；每路相机使用其自身采集时刻的自车位姿和标定；GT 从全局转换到参考车体；检测和跟踪使用项目标准化后的参考车体坐标。轨迹先在全局累计，再转换到当前参考车体，避免自车运动造成错误轨迹。

`frame` 按同一 sample 分组用于联动显示，并不代表所有传感器同时采集。相机实体保留真实时间戳和相对 LiDAR 的毫秒偏差。历史动态物体仍可能由于时间差产生错位。雷达、历史 sweeps 与除 MapTR 外的扩展模型尚未接入本查看器。

查看器读取本地记录，记录文件不上传 GitHub；服务器仅绑定本机回环地址。关闭查看器不会删除原始数据或模型结果。

## 验证记录

已通过 Rerun 自带 `rrd verify`、RRD dataframe 读取（39 个点云时刻及每路相机 39 帧），以及四项坐标适配单元测试。本机浏览器实际渲染截图保存为 `outputs/rerun/verified_viewer.png`，该目录不进入 Git。


## MapTR 车道线（2026-09-24）

已加入 39 帧真实 MapTR 预测及独立 BEV 面板；黄色分隔线、粉色人行横道、蓝色道路边界。每帧清空旧线，默认 score >= 0.5。三维层为 `ego/maptr`，二维层为 `bev/maptr`；BEV 前方朝上。模型没有高度输出，三维线绘制于近似地面 Z=0。

`tools/rerun_mini.py` 默认读取 `outputs/maptr/predictions.json`，验证摘要哈希与帧顺序。存在该文件但验证失败时拒绝导出；文件不存在时仍可导出原有场景。可用 `--maptr PATH` 指定结果，`--lane-score 0.3` 调整阈值。推理和环境说明见 [MAPTR.md](MAPTR.md)。本轮通过 18 项测试、RRD 折线逐点校验及真实 Chrome 渲染；截图 `outputs/rerun/maptr_verified.png`。更换记录后刷新页面。

BEV 车道线面板位于左上方，占左侧主要显示区域。预测线使用固定屏幕线宽，缩小视图时仍保持可见；黄色分隔线、粉色人行横道、蓝色边界。

右侧六路相机已加入 MapTR 预测折线投影，路径为各相机 `image/maptr`，可切换可见性。颜色与 BEV 一致、随帧清理更新。使用相机实际采集时刻位姿、内参和近似地面 Z=0；不做车辆遮挡判断，因此线可能绘制在车辆前景上，坡道也可能存在贴合偏差。这是 BEV 预测的几何投影，不是另跑图像车道线模型。投影与裁剪实现见 `tools/camera_overlay.py`。

## LiDARSeg 逐点颜色

新增 `--lidarseg PATH`（默认 outputs/lidarseg/predictions.json）。自动验证结果与原始点云哈希和点数；存在无效结果时拒绝导出。预测颜色在 `ego/lidar` 和 BEV 中显示；切换 **LiDARSeg ground truth** 页签查看 `lidarseg_gt/points` 官方标签。两者使用同一时间轴、相同逐点顺序。详见 [LIDARSEG.md](LIDARSEG.md)。浏览器验证截图：outputs/rerun/lidarseg_verified.png。

## CenterPoint 跟踪多视图显示（2026-09-24）

使用现有 M11 PubTracker 结果，不增加模型推理。BEV 的 `bev/tracks/<ID>` 包含旋转框、ID 标签及最近最多 20 个显示位置的轨迹；六路相机的 `image/tracks/<ID>` 显示经过近裁剪面和图像边界裁剪的三维框投影、简短 ID 标签（#编号）。三维框和轨迹也采用相同 ID 配色。颜色由 ID 确定，跨帧保持一致；相近颜色仍可能出现，以 ID 为准。

所有跟踪图层沿用 `--score`（默认 0.25），每帧清除旧实体后重绘，未输出的轨迹不会残留。轨迹点先在世界坐标累计，再变换到当前自车坐标。相机投影使用实际相机位姿，但不外推动态目标到相机曝光时刻，也不进行遮挡判断，因此快速运动或被遮挡目标可能出现投影偏差。ID 由跟踪器分配，可能因漏检或关联错误改变。

几何与配色：`tools/tracking_overlay.py`；集成：`tools/rerun_mini.py`。重新导出后刷新页面加载新记录。

## 失败案例图层

已加入检测与跟踪失败 BEV 页签、相机检测失败投影及逐帧案例统计。颜色、时间轴定位、数据校验与开关见 [失败案例回放报告](reports/rerun_failures/report.md)。默认案例阈值 0.25；修改 `--score` 时须同步报告口径或加 `--no-failures`。

## 嵌入式案例选择与自动跳转

启动 `envs/rerun/bin/python tools/start_case_browser.py`，访问 http://127.0.0.1:9092/。筛选并点击案例后自动暂停并定位 frame，联动更新 BEV 和六路图像。完整记录就绪前暂不允许跳转，防止定位到未加载数据。见 [操作与复现](reports/embedded_case_browser/report.md)。

## 事件归并与人工复核（2026-09-29）

9092 默认事件工作台。切换“问题分组排行榜”→查看组内事件→展开原始记录；“播放前后2秒”按采样时间回放并在末帧暂停。原始案例模式与旧链接继续可用。

事件/分组详情下方可填写负责人、调查记录、措施与验证证据，保存后刷新保持；复核记录在 outputs/case_browser/reviews/ 按数据版本隔离，可导出快照。已解决必须填写验证说明。见 [结果、规则和验证](reports/failure_events/report.md)。

## 异常优先图层（2026-09-29）

9092 默认加载 triage_scene.rrd，仅显示失败框与背景；顶部“显示正常目标”首次勾选才加载约4.5 MiB附加图层，取消隐藏。事件选择自动切换检测/跟踪分支，包括六路相机和3D，不再需要手动切换失败页签。见 [规则、生成命令和验证](reports/failure_display/report.md)。
