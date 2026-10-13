# HARP-NeXt：资源受阻，mini 预处理已实测

状态 **blocked_resource**。官方源码固定到 `532df2659700ae870c0e0d4137a4d35bcc9903b9`，原始点审计、mini 数据入口、CPU/GPU 预处理诊断完成；完整 nuScenes-32×480 checkpoint 未下载。严格加载、真实模型 forward、模块 Hook、39/81/404 帧模型推理、mIoU、预测回放均未完成。没有随机权重或替代网络结果。

## 数据与 GT 隔离

本次逐文件重新核验 mini 404 帧、14,026,208 个原始点与同数量标签，保存点云/标签 SHA256 和样本 token。scene-0061 为39帧，mini_val 为81帧。生成三个与作者 `nuscenes_files.npz` 相同三列 schema 的 val 列表（点云路径、标签路径、sample_data token），使用真实长度，未读取 full-val6019列表。作者映射文件与本工具32→16类表逐项相等，网络ID0–15、显示ID1–16，ignore255。

首/中/末 CPU 预处理分别将34,688/34,688/34,720原始点扩展到53,638/53,291/54,923点。原始点始终是完全相等的前缀。实际GT、全零标签、人工替换标签三种输入产生完全相同的 CPU 网络特征和投影坐标，GT投影标签未作为网络输入。预测级 GT 独立性因缺权重尚未检验。作者 manager 消费处理后点标签，包含这些追加点；公共评价辅助函数明确恢复原始点顺序并排除追加点，缺类 IoU 留空。

## GPU 候选结果与决定

单张 RTX4090 串行测试，batch1、FP32；实际 CUDA torch-scatter smoke 得到预期 `[2,8]`。采用作者原始 CPU/GPU 算法，只对 Torch2.1 不支持的两个空 uint32 初始化缓冲做兼容补丁，详见 patch。没有更改投影算法。

| 预处理路径 | 均值 ms | P50 ms | P95 ms | 标准差 ms |
|---|---:|---:|---:|---:|
| CPU |65.782|63.531|74.328|8.589|
| GPU |3.107|3.137|3.525|0.251|

同一首帧预热3轮后30轮交错配对，计时为暖缓存原始文件读取→预处理→H2D→voxel/coor回传，前后同步；不含模型、GT读取或可视化。峰值 GPU allocated 为30,810,112字节。这些数字不是模型端到端 FPS。

**不采用 GPU 候选。** 三帧 GPU 均比 CPU 多15个合成点；原始点投影坐标差异分别为0/1/0点。重复GPU运行的特征不完全一致，GT替换比较也不完全一致，因此后者受到非确定性混杂，不能据此断言标签泄漏，也不能宣称隔离通过。源码的重复索引 scatter 写入可能影响碰撞点选择，尚需独立定位。CPU排除水平边界插值而GPU `torch.roll` 环绕；专门构造的三点边界样例实测CPU输出3点、向量化实现输出4点。两者插值 occupancy 都用 `proj_idx>0`，点索引0会被视作空像素；相同三点几何仅调换点顺序，CPU插值输出从3点变为4点，实测确认该影响。未静默修正这些模型输入语义。

## 资源与限制

[作者仓库](https://github.com/SamirAbouHaidar/HARP-NeXt)已下载；[作者权重分享](https://cloud.minesparis.psl.eu/index.php/s/eUOcaEXZRLNfV1e)可偶尔返回HTML页面，但多次WebDAV和AJAX目录请求超时。猜测目录名的下载探测返回8,198字节 `File not found` HTML，已拒绝作为checkpoint。权重文件名/大小/hash均未知，无模型指标。机器环境与作者Torch2.5.0不同，只确认了本次预处理与scatter操作，不能外推整个网络兼容。

该阶段是 nuScenes mini 教学诊断；与预训练数据可能重叠。作者论文精度和FPS未填作本机成绩。已实现原始点评价函数、mini预处理与真实模型适配：作者Network构造→checkpoint['net']严格加载（仅允许统一module前缀）→ConvSENeXt/融合/head Hook→首中末预测级GT独立门控→39/81/404点级评价与预测保存。该模型分支因缺权重尚未实跑，状态为untested；当前 `--phase run` 返回资源阻塞状态及退出码2。

## 复现

从仓库根目录运行；原始数据和上游源码/环境留在本地，不发布二进制资产。

```bash
# 克隆并固定源码后，Torch2.1环境需应用仅空标签缓冲兼容补丁。
git -C perception_lab/third_party/HARP-NeXt_2025 apply ../../reports/perception_2025_20261013/harpnext/torch21_empty_label_compat.patch
perception_lab/envs/harpnext_2025/bin/python perception_lab/tools/harpnext_2025.py --phase preflight
# GPU必须单独运行，避免与其他模型同时计时。
perception_lab/envs/harpnext_2025/bin/python perception_lab/tools/harpnext_2025.py --phase audit_gpu
perception_lab/envs/harpnext_2025/bin/python perception_lab/tools/harpnext_2025.py --phase run --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/harpnext_2025/bin/python -m unittest discover -s perception_lab/tests -p test_harpnext_2025.py -v
```

4项针对性测试通过：完整checkpoint结构及统一DDP前缀、类映射与ignore、含追加点/乱序点的恢复及重复/漏点拒绝、有效GT与缺类指标口径。详细数据见 `preflight.json`、`mini_frames.json`、`gpu_preprocessing_audit.json`、`resource_manifest.json`。
