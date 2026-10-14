# 浏览器初始化故障定位与兼容路径

本轮没有把浏览器功能验收计为 GPU 模型/算子性能。软件失败与兼容运行都保留证据。

## 可复现的失败边界

1. 已安装 Playwright 版本在 `headless=True` 时添加 `--headless=old`；当前 Google Chrome 已移除该模式。启动日志直接报 `Old Headless mode has been removed`，浏览器尚未加载页面。脚本改为显式 `--headless=new`，避免旧参数注入。
2. 指定 SwiftShader 后，WebGL 实际 renderer 是 `ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)`。其 `MAX_COLOR_ATTACHMENTS` 和 `MAX_DRAW_BUFFERS` 均为 **6**。Rerun 0.23.4 的设备创建要求 **8**，页面及 console 均报 `Limit 'max_color_attachments' value 8 is better than allowed 6`。因此不能把这条软件路径宣称通过。
3. [短诊断收据](swiftshader_initialization_diagnosis.json)显示 handle 已构造，但 recording_id、timeline、frame range 全为空；panic 为 false，JS heap 约 41 MB。Viewer **尚未请求 RRD**，排除“RRD 大小导致流加载超时”的解释。同一浏览器独立 fetch RRD 得到 HTTP 200、84,137,275 bytes、SHA-256 `2da4caecb87ea4aca9f7e7e6d473a08f5fdac7f6eda2f92bd61803fde5bdacc1`，与磁盘实际导出完全相符。[失败时页面截图](swiftshader_initialization_failure.png)可见具体初始化错误。

两次更早的 headful 超时保存在 [headful 失败收据](browser_attempt_20261014_headful_failed.json)；这些超时没有被独立确证为同一个根因，不用后来的 SwiftShader 错误倒推解释。

## 已验证的兼容运行方式

[Chromium 官方说明](https://chromium.googlesource.com/chromium/src/+/refs/heads/main/docs/gpu/using-gpu-hardware-in-headless-chrome.md)明确：headless 必须加 `--enable-gpu` 才不强制软件渲染。脚本使用 `--headless=new --enable-gpu`，并在等待 recording 之前检查实际 renderer；硬件请求若意外落到 SwiftShader 立即失败，避免再等待无意义的 recording predicate。

本机该路径实际为 `ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 4090/PCIe/SSE2, OpenGL 4.5.0)`，scene0061 的 12 个模型页/帧状态通过，未出现页面错误、未解释应用错误或 panic。mini_val 的验收状态与跨场景证据以其独立 `mini_val/rerun_browser_qa.json` 为准。

运行时存在外部训练；检查时空闲显存 19,474 MiB / 总量 24,564 MiB，未更改或终止其他进程。这里只验证页面功能与截图，不记录 GPU 吞吐、时延或声称独占资源。参数、renderer 和 frame state 均保存在最终 QA 收据；失败尝试另存 `failed_attempt_*.json`。

复现故障证据：`python perception_lab/tools/tracking_gpu_browser_diagnose_20261013.py`。复现兼容验收：`python perception_lab/tools/tracking_gpu_rerun_qa_20261013.py --headless`，mini_val 添加 `--split mini_val`。软件模式保留供明确诊断，不默认视为兼容配置。
