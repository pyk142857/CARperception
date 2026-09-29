# 完整点云与 3D 目标取景

日期：2026-09-29。

## 行为

选中异常事件或事件成员时，3D 相机注视目标中心并拉近，当前帧的全部 LiDAR 点仍保留在视图中。没有空间裁剪或新增降采样；可以手动缩小、旋转查看周围场景。这里的“全部”指当前帧原始 LiDAR 扫描，不是将历史 39 帧累积叠加。

3D 视图始终包含 `/ego/lidar` 和完整车道线 `/ego/maptr/**`。仅显示选中的异常框；正常框仍通过复选框加载完整正常目标层。BEV 与相机图像继续按目标取景。

本报告取代 [此前局部点云取景方案](../selection_3d_focus/report.md) 中的点云裁剪行为。

## 实现

- `tools/selection_context.py` 仅计算相机：注视目标中心，观察方向为归一化的 `[-0.7, -1, 0.65]`，距离为 `max(6 m, 2.5 × 目标尺寸对角线长度)`。
- `tools/triage_views.py` 使用 `EyeControls3D(position, look_target, eye_up=[0,0,1])`；选择目标不改变点云实体或坐标。
- `tools/selection_blueprint.py` 不再生成局部点云和局部车道线副本。
- 工作台 Web Viewer 和独立 `envs/rerun_focus` 使用 0.27.3；原导出环境 0.23.4 和已有录制保持兼容，浏览器实测通过。JS 与 WASM 使用同一版本目录，包含无扩展名 JS 路由的回归测试。

相机参数定义见 [Rerun 0.27.3 官方源码](https://github.com/rerun-io/rerun/blob/0.27.3/rerun_py/rerun_sdk/rerun/blueprint/archetypes/eye_controls3d.py)。

## 验证

- [点云审计](pointcloud_audit.json)：39 帧共 **1,354,112 点**，每帧录制数量与原始 LiDAR 完全相同；外参变换后的最大坐标差为 **0.000003815 m**，来自存储精度。
- [61 项测试通过](tests.txt)：包括完整点云实体选择、相机取景与资源版本一致性。
- [浏览器验证](verification.json)：初始选择、切换事件成员、正常框开关、片段播放、清空选择、快速切换、FN 选择均通过；无页面异常，主场景仅请求一次。
- [帧 2 截图](selected_frame2.png)、[帧 3 截图](selected_frame3.png)、[FN 截图](fn_selected.png)：完整点云背景与目标放大并存。

点云审计验证录制内容；实体路径测试验证选中视图使用该完整点云；浏览器验证相机取景与交互。未重新推理或修改评估结果。

## 复现

在项目根目录执行，需已有 mini 数据与录制文件：

```bash
perception_lab/envs/rerun/bin/python -m venv perception_lab/envs/rerun_focus
perception_lab/envs/rerun_focus/bin/python -m pip install -r perception_lab/requirements-rerun-focus.txt
python3 perception_lab/tools/start_case_browser.py --stop
python3 perception_lab/tools/start_case_browser.py --no-browser
perception_lab/envs/rerun_focus/bin/python perception_lab/reports/full_pointcloud/audit.py
perception_lab/envs/mmdet3d/bin/python -m unittest discover -s perception_lab/tests
/usr/bin/python3 perception_lab/tests/browser_selection.py --out /tmp/car_fullcloud_check
```

打开或刷新 <http://127.0.0.1:9092/>。浏览器测试需要 Playwright、Pillow、Google Chrome 和图形桌面；运行环境、原始数据及大型 RRD 不上传 GitHub。

后续修正：[恢复深色主题](../dark_theme_restore/report.md)，完整点云与取景行为不变。
