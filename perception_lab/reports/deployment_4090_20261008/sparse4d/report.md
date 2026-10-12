# Sparse4D v3 R50：真实六相机时序检测与跟踪

实测日期 2026-10-12。S4 推理与官方代码评估已完成：scene-0061 全39帧、官方 mini_val 两场景全81帧。以下是 mini 教学诊断，不是全量 nuScenes val 榜单；mini/train 场景可能与预训练数据重叠。模型常驻，六相机256×704，batch1，使用作者FP16包装及float32自定义聚合，原生900实例、历史600实例。推理不读取点云，不使用GT。

|范围|帧|mAP|NDS|AMOTA|AMOTP / m|IDS|
|---|---:|---:|---:|---:|---:|---:|
|scene0061|39|0.584579|0.545315|0.679487|0.777958|9|
|mini_val|81|0.461608|0.433072|0.637567|0.819554|16|

DetectionEval/detection_cvpr_2019、TrackingEval/tracking_nips_2019，devkit1.2.0原代码。检测保留300个模型解码框，无渲染阈值过滤；属性为空，因为网络未预测属性，也未采用作者速度规则补属性，因此NDS的属性分项受影响。跟踪采用作者 classification score≥0.2、官方7类，保留原生ID并增加scene_token命名空间。没有与旧单场景CenterPoint指标混比。

|范围|模型含解码D2H P50/P95 ms|真实端到端 P50/P95 ms|峰值allocated/reserved MiB|
|---|---:|---:|---:|
|scene0061|24.73 / 25.72|219.34 / 223.98|470.2 / 702.0|
|mini_val|24.46 / 25.66|217.02 / 224.77|470.2 / 702.0|

端到端包括六图读取/测试预处理、H2D、模型+作者解码/D2H、标准化输出；不包含JSON最终落盘、视频和离线评估。逐帧CSV分列这些边界；内部模型/解码/D2H融合在一次调用内，未声称是纯kernel时间。独立预热3次，每次reset，正式连续帧保留记忆。具体分段均值见各split的summary.json；CPU图像读取和预处理是本轮的主要开销。只表示RTX4090桌面GPU。process_gpu_memory_snapshot.csv为运行时nvidia-smi快照，不冒充进程峰值。

## 坐标、输入与时间状态核验

采用固定官方commit的mini converter与原测试pipeline：原1600×900图等比0.44缩放到704×396，再裁剪底部256行，无翻转/随机旋转，RGB归一化；相机顺序见各summary。原生输出为LIDAR_TOP重心框，尺寸按作者官方导出器[4,3,5]变成wlh，再经本帧LIDAR calibrated_sensor变到ego。六相机分别使用实际标定与各自ego pose，而非拿一个时间点替代全部相机。

首/中/末3帧×6相机，以元数据独立重算异步camera←global←ego←lidar矩阵，与作者转换器比较；原生→ego→global和回转核验见input_coordinate_audit.json、coordinate_audit.json。非平凡90°外参单元测试通过。

每个场景起始显式InstanceBank.reset()并清理dn_metas，缓存为空、prev_id=0；ID外层命名空间按scene隔离。保存第2帧后全部8项InstanceBank状态到磁盘、重置、恢复后续第3帧，验证boxes/scores/类别/ID；独立重置后重跑第一帧也验证。IDs与类别完全一致。作者CUDA forward使用atomicAdd，数值不保证bitwise一致：最初1e-5检查失败，保留该事实，现以box绝对误差<1e-3、score<1e-4且ID/class精确一致判定数值等价，实测差值逐项保存在state_tests.json。没有将误差归零或写成bitwise通过。

首次完整部署与最终CLI复跑的官方指标也存在轻微变化，详见[整段复跑对照](rerun_repeatability.json)：不仅框浮点值不同，原生跟踪ID相关指标也可能变化。上游atomic累加已确认存在，但尚未隔离所有整段变化来源，不能承诺39/81帧整段逐位确定性。本文表格和压缩预测均采用最后一次CLI验证结果。

## 环境、权重与来源

官方源码commit `249ffbb695f4e9db628d953e2bf6d36de04bbb69`，配置sparse4dv3_temporal_r50_1x8_bs6_256x704.py；官方R50 release权重strict=True加载，无缺失/多余key。官方checkpoint已含训练后的head.instance_bank.anchor[900,11]，直接提取该精确Parameter作为构建prior，随后严格加载整个state_dict；未在mini_val重新拟合kmeans。URL、大小与SHA256见artifacts_manifest.json，逐split strict_weights.json含prior哈希。

独立envs/sparse4d仅通过.pth只读复用maptr依赖，不修改旧环境。CUDA11.8编译native SM89；PyTorch1.13的自动架构列表无8.9，sm89.patch显式加入nvcc gencode。CUDA11.8 toolkit与Torch cu117存在minor version警告，真实推理已验证。MMCV配置dump需要作者yapf0.33，本地覆盖安装；官方评估使用nuscenes1.2.0以支持自定义split，另安装motmetrics/xmltodict；完整freeze与编译日志保留。

## 产物和局限

各split目录有标准化与原生预测gzip（完整帧、完整300框，不是下载预测）、官方metrics_summary/details、逐帧timings、实际配置、权重/坐标/状态核验。大权重/转换pkl/状态文件在outputs、checkpoints本地，路径和哈希见manifest。scene0061/failure_cases.json给出最多5个带token、距离、模型分数证据的FN；诊断为score≥.25、同类2m Hungarian，与官方AP和IDS定义不同；根因仅列假设。同步回放/代表图由本轮统一visualize阶段提供。

本机完整trainval val未评估；未微调；未宣称车端实时性或提供跨split模型优劣结论。
