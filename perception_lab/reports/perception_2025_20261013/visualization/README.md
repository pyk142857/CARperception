# 2025 实测回放验收

本次回放覆盖 scene-0061 的 39 个关键帧。GRAE-3DMOT 使用封存参考检测输入，DEIMv2-S 为 234 张六相机实测预测；原始 LiDAR 共 1,354,112 点。MambaFusion / HARP-NeXt 缺失预测明确显示缺失，未复用其他模型或上一帧填充。GT 为独立 Rerun 图层，未用于生成预测。

显示阈值为 0.25，不修改评价预测文件。坐标为 LiDAR 时间戳的 ego FLU，各相机使用自己的位姿；轨迹先存 global，再变换到当前 ego。ID 颜色使用固定 ID 映射；轨迹跨漏检间隔重新开始。逐帧清除模型、BEV、GT 和六相机子树，避免失踪框/轨迹/2D 框残留。首、中、末帧对应 0、19、38。

`status.json` 保存数据来源 SHA256、逐帧 token/点数/框数、导出状态及浏览器状态。导出后重新读取 RRD，检查所有 39 帧 token 与预测框数、全部原始点坐标、351 次递归清除记录；浮点容差为 1e-4 m，实际值见 `decoded_rrd_audit`。`frame_00.jpg`、`frame_19.jpg`、`frame_38.jpg` 为预测代表图：橙色为 DEIMv2 2D 框，其他颜色为 GRAE ID。

复现（从 `perception_lab/` 执行）：

```bash
envs/rerun/bin/python tools/perception_2025_visualize.py --config configs/perception_2025.yaml
# GPU 推理/计时结束后，独占 GPU 运行以下浏览器验收；复用 9094 / 9095 服务。
python tools/perception_2025_rerun_qa.py
```

本地入口：http://127.0.0.1:9094/perception_2025/ 。完整 RRD 和 39 张逐帧图位于被忽略的 `outputs/rerun/deployment_4090/perception_2025/`，报告保留代表图及复现方式。交互回放需要该本机数据与服务，仓库中代表图可独立查看。

Browser plugin not available，因此使用已安装的 Playwright / Chrome。浏览器检查目标：真实 Rerun 加载 → 暂停 → 切换 frame 时间轴至 0/19/38 → API 读回对应时间、无 panic、画面渲染；同时检查轻量预览滑块更改 token/图像。浏览器结果及截图在 `rerun_browser_qa.json` 和 `rerun_browser_*.png`。手机尺寸仅检查轻量预览，Rerun 的高密度桌面布局不作手机适配通过声明。截图不代表官方全量评价；当前仅展示成功产出的两项模型。

## QA 核验范围

| 检查 | 方法与依据 |
|---|---|
| 页身份与非空 | Rerun Viewer 标题、精确 localhost URL、canvas、实际渲染截图 |
| 时间轴交互 | WebHandle 设置并读回 frame 0/19/38、范围 [0,38]、暂停状态、无 panic |
| 坐标/点数 | 从保存的 RRD 重新解码全部点，与原始 LiDAR→ego 结果比对 |
| 图层清理 | 解码检查每帧 models/BEV/GT/六相机的递归 Clear |
| GT 独立 | 单独实体与 Tab；点击 Independent GT 并截图 |
| 控制台 | page errors、console warnings/errors、失败请求 URL 均保留，不能用无 pageerror 代替无 console error |
| 手机尺寸 | 390×844 轻量预览；token 换行且图片缩放，Rerun 仅桌面验收 |

首次浏览器核验发现默认选中 guide 页签及手机 token 溢出，现已修改默认 predictions 页签和手机换行。3D 视图只展示预测与 LiDAR，避免相机 pinhole 可视化警告；六相机标定与投影在独立 2D 视图中保留。以最新 `rerun_browser_qa.json` 的状态、RRD hash、控制台及截图作为验收依据。

最终验收通过：1800×1100 的真实 RTX 4090 WebGL2 回放，frame 0/19/38、独立 GT 页签、390×844 预览均完成并逐张检查截图；无 page error、panic 或框架错误遮罩。将 blueprint 提前到帧数据之前后，启动时 pinhole 错误消失。最新控制台仅有已分类的外部 Google 字体超时及示例目录预取取消，未影响 localhost RRD；完整请求 URL 和原始日志保留于 JSON，不声称控制台完全无警告。三帧显示框数为 19/37/22，原始点数为 34,688/34,688/34,720。
