# 冻结 BEVFusion / GRAE / OptiPMB 同步回放

独立入口为 `http://127.0.0.1:9094/tracking_gpu_20261013/`，mini_val 为其 `mini_val/` 子目录。原有 2025 感知与部署 RRD 未修改。两套新 RRD 的全部帧已完成实际解码核验，兼容配置下的真实浏览器首/中/末与跨场景功能验收均通过。

## 实际导出

| 分割 | 帧数 | 原始点数 | BEV 显示框 | GRAE 显示框 | OptiPMB 显示框 |
|---|---:|---:|---:|---:|---:|
| scene0061 | 39 | 1,354,112 | 2,634 | 1,189 | 1,136 |
| mini_val | 81 | 2,812,480 | 3,238 | 3,174 | 3,022 |

显示阈值统一为 score ≥ 0.25，不改变各作者原生预处理、跟踪输出与官方评价输入。原始 BEVFusion 检测、GRAE、OptiPMB 和独立 GT 各有分页，包含 3D、BEV、六相机。原始点云全点加载；跟踪颜色由算法前缀与 ID 决定，轨迹跨场景与断点重置。GT 是独立评价/展示读取，不进入跟踪器输入。GT 展示页包含原始标注；下方教学统计另按官方类别、距离、点数与 rack 规则过滤。

输入检查包含冻结检测/packet hash、模型当前完成状态、tracking receipt、官方 evaluation_status 的 tracking/metrics hash、protocol 的检测/packet/GT hash 与完整 token 顺序。点云原文件 hash 与 LiDAR→global 变换也核对。相机投影使用相机各自时刻的 ego pose。每帧清空模型、BEV、GT 与六相机层，避免旧图层残留。

[39 帧解码证据](decoded_audit.json)与[81 帧解码证据](mini_val/decoded_audit.json)核对每个 sample_token、原始点坐标、模型框中心/数量、六相机投影线段数与递归清除。最大原始点 float32 坐标误差约 3.815×10⁻⁶ m，框中心误差小于 1.91×10⁻⁶ m。完整源文件 hashes 与逐帧状态在各 `status.json` 中。

## 固定教学案例

[案例图片索引](teaching_cases.md)保留逐例 token 与两方法对照图；BEV 红圈为真实后验 FP，橙叉为 FN。[teaching_cases.json](teaching_cases.json)使用冻结预测后的独立 GT 匹配：score ≥ 0.25、同类 XY 距离 < 2 m、先保留上帧有效对应再 Hungarian；官方 devkit 的类别距离/点数/rack 过滤；不做官方轨迹插值。**教学 IDSW 与官方 IDS 不同**，以下数字不替代 AMOTA/AMOTP。

| 分割 / 方法 | TP | FP | FN | 教学 IDSW |
|---|---:|---:|---:|---:|
| scene0061 GRAE | 1,030 | 20 | 186 | 5 |
| scene0061 OptiPMB | 1,007 | 16 | 209 | 6 |
| mini_val GRAE | 2,821 | 205 | 537 | 12 |
| mini_val OptiPMB | 2,737 | 159 | 621 | 7 |

选例规则先固定：每个分割对 FP、FN、IDSW 分别取 Opti−GRAE 最大严格改善和最大严格退化，绝对差相同取最早帧；不存在某方向则不选图。scene0061 没有任何 FN 改善帧，因此没有生成 FN 改善图。每例保留 frame、sample_token、两方法计数、差值与图片路径；案例是描述性分析，不调参、不外推泛化提升。

实际所选 scene0061：FP 改善 8、FP 退化 9、FN 退化 0、IDSW 改善 17、IDSW 退化 3。mini_val：FP 改善 29、FP 退化 48、FN 改善 52、FN 退化 40、IDSW 改善 2、IDSW 退化 1。每例有 GRAE / OptiPMB 同帧六相机 + BEV 对照图片。

官方本轮结果另见模型的 bound evaluation：scene0061 AMOTA GRAE 0.862569 / Opti 0.888718，mini_val 0.822964 / 0.797533。Opti 在 scene0061 改善而 mini_val 退化；教学阈值下 FP 较少同时 FN 较多，不描述为全面改善。GPU 版在两个分割的 tracking.json 字节完全相同，见[输出等价证据](../grae_gpu/equivalence.json)；[官方指标等价](../grae_gpu/official_equivalence.json)也通过，因此沿用当前 GRAE 回放，不重复绘制相同内容。

## 浏览器状态与复现

最终使用 `--headless=new --enable-gpu`，实际 renderer 为 ANGLE / NVIDIA RTX 4090 / OpenGL 4.5。39 帧检查 0/19/38 × 4 页，共 12 个模型页/帧状态；81 帧检查 0/39/40/80 × 4 页，共 16 个状态，包含 39→40 的跨场景转换。两次均无 page error、未解释应用错误或 panic，Chrome 均已关闭。存在外部训练时仅验证功能，没有进行图形/模型/算子性能测量，未改动其他进程。

[39 帧浏览器收据](rerun_browser_qa.json)、[81 帧浏览器收据](mini_val/rerun_browser_qa.json)保存页面身份、frame API、截图和 renderer。代表性截图：[GRAE 19](rerun_grae_19.png)、[Opti 19](rerun_optipmb_19.png)、[Opti 跨场景前 39](mini_val/rerun_optipmb_39.png)、[跨场景后 40](mini_val/rerun_optipmb_40.png)、[GT 80](mini_val/rerun_ground_truth_80.png)。两次 overview 滑块检查与 390×844 截图也完成；密集 Rerun 工作区仍以桌面为验收范围。

**截图底栏的 30 FPS 是 Rerun 播放设置，不是模型推理吞吐率；nuScenes keyframe 约为 2 Hz。** 可切换 `elapsed` 时间线或调整播放速率；记录保留每帧真实 timestamp_us 与 elapsed 秒值。

初始化故障的分层证据与修复见[根因说明](browser_root_cause.md)：旧 Playwright 注入已移除的 old-headless 参数；默认/显式 SwiftShader 仅支持 6 个颜色附件，而该 viewer 要求 8。失败页、handle 空状态、完整 RRD 浏览器 fetch hash 与失败 receipts 均保留。SwiftShader 路径没有被宣称通过；早期 headful 超时也没有被倒推为已确证的同一原因。

仓库根目录命令：

```bash
python -m pytest perception_lab/tests/test_tracking_gpu_visualize_20261013.py -q
perception_lab/envs/rerun/bin/python perception_lab/tools/tracking_gpu_visualize_20261013.py
perception_lab/envs/rerun/bin/python perception_lab/tools/tracking_gpu_visualize_20261013.py --split mini_val
perception_lab/envs/sparse4d/bin/python perception_lab/tools/tracking_gpu_cases_20261013.py
python perception_lab/tools/tracking_gpu_rerun_qa_20261013.py --headless
python perception_lab/tools/tracking_gpu_rerun_qa_20261013.py --split mini_val --headless
```

三项 CPU 门禁测试已通过，上述导出、案例与最终浏览器命令均已实际执行。大 RRD 保存在 ignored 输出目录，报告保留可重建来源与 hashes；原始数据、模型权重不提交。
