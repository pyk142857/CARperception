# 2025 原生结果同步回放验收

scene-0061 的 39 帧与 mini_val 的 81 帧均已重新导出并通过 RRD 解码核验、真实浏览器首/中/末帧与模型分页检查。数据来自本轮实际 MambaFusion、GRAE-3DMOT 两种输入族、HARP-NeXt 的已完成推理产物。该结论是 mini 教学回放验收，不是官方全量模型评测，也不消除模型报告中记载的数值稳定性或计时限制。

## 数据与显示口径

| 分割 | 帧数 | 原始 LiDAR / HARP / 独立 GT 点数，各层 | Mamba 显示框 | GRAE frozen 显示框 | GRAE native 显示框 | DEIM 图像对 |
|---|---:|---:|---:|---:|---:|---:|
| scene0061 | 39 | 1,354,112 | 2,976 | 1,189 | 1,403 | 234 |
| mini_val | 81 | 2,812,480 | 3,827 | 3,174 | 3,763 | 0 |

框的显示阈值为 score ≥ 0.25；不裁剪用于模型评估的完整预测。DEIMv2 沿用本轮实测的 scene0061 六相机 234 对输出；mini_val 未生成 DEIM 2D 输出，六路原始相机照片和 3D 框/语义投影仍完整显示，不用其他帧代替。

每页包含 3D、BEV 和六相机：MambaFusion、GRAE frozen_reference、GRAE mambafusion_native、HARP-NeXt、独立 GT。DEIM 相机框是橙色叠加层。检测按类别固定颜色，跟踪按带输入族前缀的 track_id 固定颜色；历史轨迹跨场景或帧断点清空。HARP 与 GT 分页不叠加同位置的灰色点云，避免遮挡语义颜色。

全局检测和轨迹转到当前 LiDAR 时间的 ego 坐标；点云由原始 LiDAR 坐标经外参变换。相机变换包含其独立采样时刻的 ego pose。HARP 的每个原始点索引、总数与 NPZ hash 均检查，GT 标签仅作为独立可视层读取。模型、BEV、GT 与相机子树逐帧递归清除；缺失层为空。

## 证据

- [39 帧状态与源文件 hashes](status.json)、[逐帧计数](frame_counts.csv)、[解码核验](decoded_audit.json)、[浏览器 QA](rerun_browser_qa.json)。
- [81 帧状态与源文件 hashes](mini_val/status.json)、[逐帧计数](mini_val/frame_counts.csv)、[解码核验](mini_val/decoded_audit.json)、[浏览器 QA](mini_val/rerun_browser_qa.json)。
- [39 帧原生 MambaFusion + DEIM 短片](native_mambafusion_scene0061.mp4)、[视频探测与 hash](video_manifest.json)：1920×1080，H.264，39 帧，固定 2 fps，19.5 秒。视频是离线显示，不代表推理吞吐率；精确传感器时间戳保存在 RRD 与逐帧状态中。
- [39 帧 Mamba 中帧](rerun_browser_19.png)、[native GRAE 中帧](rerun_grae_mambafusion_native_19.png)、[HARP 首帧](rerun_harpnext_00.png)、[中帧](rerun_harpnext_19.png)、[末帧](rerun_harpnext_38.png)、[独立 GT](rerun_ground_truth_19.png)。
- [81 帧 Mamba 中帧](mini_val/rerun_browser_40.png)、[native GRAE 中帧](mini_val/rerun_grae_mambafusion_native_40.png)、[HARP 首帧](mini_val/rerun_harpnext_00.png)、[中帧](mini_val/rerun_harpnext_40.png)、[末帧](mini_val/rerun_harpnext_80.png)。

RRD 实际解码检查了所有帧的 sample_token、框数、完整点数与 ego 点坐标、递归清除记录，以及 HARP/GT 六相机实际投影点数。浮点存储最大坐标误差：39 帧 3.81445×10⁻⁶ m，81 帧 3.81450×10⁻⁶ m；每层点数与原始点云完全一致。源模型产物只有通过当前严格加载/推理/评估状态与适用 receipt hash 检查才导入，文件存在本身不视为有效结果。

浏览器使用现有 Playwright + Google Chrome（当前无 Browser 插件 skill），1800×1100，Rerun SDK 0.23.4 / WebGL2 / RTX 4090。每个分割检查 3 次主页面帧状态和其余 4 页各 3 次帧状态，即每个分割 15 个模型页/帧状态。39 帧为 0/19/38，81 帧为 0/40/80。两次 QA 均无 page error、无未解释应用错误、无 panic；截图可见各新模型与独立 GT 的差异。overview 页面另检查滑块与 390×844 布局。浏览器已关闭，未停止其他 GPU 进程。

可视限制：3D 自动视野保留全范围点云，远点会使中心目标显示较小，可交互缩放；相机保持原始宽高比并有留黑。六相机投影是标定后的几何叠加，不包含相机深度遮挡求解。密集 Rerun 工作区以桌面为验收对象，手机仅验证轻量 overview。

## 复现

在仓库根目录运行；输入数据/权重不随 Git 分享，来源与 hash 见各模型报告。9094/9095 复用项目已存在的本地服务。RRD 大文件在 `perception_lab/outputs/rerun/deployment_4090/perception_2025/`，由以下命令重建，Git 保留报告、截图和短片。

```bash
python -m pytest perception_lab/tests/test_perception_2025_visualize.py -q
perception_lab/envs/rerun/bin/python perception_lab/tools/perception_2025_visualize.py
perception_lab/envs/rerun/bin/python perception_lab/tools/perception_2025_visualize.py --split mini_val
python perception_lab/tools/perception_2025_rerun_qa.py
python perception_lab/tools/perception_2025_rerun_qa.py --split mini_val
ffmpeg -hide_banner -loglevel error -y -framerate 2 -i perception_lab/outputs/rerun/deployment_4090/perception_2025/frame_%02d.jpg -frames:v 39 -c:v libx264 -threads 2 -preset fast -crf 25 -pix_fmt yuv420p -movflags +faststart perception_lab/reports/perception_2025_20261013/visualization/native_mambafusion_scene0061.mp4
```

浏览器验证会使用 GPU，按项目约定与模型计时串行执行。当前 6 项 CPU 测试通过，覆盖缺失产物、不完整点映射、全局/ego/相机几何、失败/运行状态门禁、HARP 完成状态及 native GRAE 输入 hash 一致性。
