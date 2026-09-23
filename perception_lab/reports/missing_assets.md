> 范围更新（2026-09-23）：BOSS 已改为 mini 教学目标。本文所列全量数据与扩展模型资产不再阻塞当前教学闭环，保留供扩展时参考。参见 [学习说明](../LEARNING_GUIDE.md)。

# 尚缺资产与资源（持续更新）

状态：partial。仅下载/导入通过不代表任务推理、评估或部署通过。

| 资产/资源 | 用途 | 官方入口 | 本地期望位置 | 当前状态 |
|---|---|---|---|---|
| nuScenes trainval metadata、samples、sweeps | 正式 val 回放、3D/深度/地图评估、80 场景训练 | https://www.nuscenes.org/download | data/nuscenes/v1.0-trainval/{scene,sample,sample_data,ego_pose,calibrated_sensor,sample_annotation}.json，以及对应 samples/、sweeps/ | 未发现本地数据；需账号许可及足够存储；归档各分片名称需从登录后的官方页面核实，不伪造 |
| nuScenes map expansion | MapTR 地图标签 | https://www.nuscenes.org/download | data/nuscenes/maps/expansion/{boston-seaport,singapore-hollandvillage,singapore-onenorth,singapore-queenstown}.json | 尚缺 |
| Cityscapes leftImg8bit_trainvaltest.zip、gtFine_trainvaltest.zip | SegFormer 官方 val 评估 | https://www.cityscapes-dataset.com/downloads/ | data/cityscapes/leftImg8bit/val/、data/cityscapes/gtFine/val/ | 官方页面重定向登录；未提供账号许可后的文件；无需在聊天中发送密码 |
| SurroundOcc 语义 checkpoint | M10 推理 | https://pan.baidu.com/s/1179t83Z5wFNNnxnPeo6n1A?pwd=dmcq | checkpoints/surroundocc/ | 作者百度网盘链接；准确文件名需解析下载内容，尚未核实 |
| SurroundOcc val infos | M10 配套输入 | https://pan.baidu.com/s/1vbDe1FtW-ThDv21KDjJ6Ig?pwd=e81b | data/surroundocc/nuscenes_infos_val.pkl | 尚缺 |
| SurroundOcc 200x200x16 val occupancy，0.5m | M10 官方评估 | https://pan.baidu.com/s/1UgiGm-ftrA91QBuEgmauTQ?pwd=31y8 | data/surroundocc/nuscenes_occ/ | 尚缺；不使用 Occ3D 代替 |
| GPU 可用时段 | 重型模型、正式性能对比、TensorRT 和训练 | 本机 RTX 4090 | GPU 0 | 已有用户任务占用约17GB显存、GPU利用率90–100%；未终止这些进程，不能取得独占性能结果 |
| 更大数据盘或已准备数据引用 | 全量数据与模型缓存 | 本机挂载路径 | 待提供 | 初检系统盘仅余74GB；当前只准备mini与COCO，不删除用户数据 |

官方模型/占用来源已固定在 configs/locked/repositories.json；资产连接原始证据见 logs/legacy_asset_probe.json。

辅助资产补充核查：常见 `https://www.nuscenes.org/data/can_bus.zip` 和 `.../nuScenes-map-expansion-v1.3.zip` 地址在当前请求中返回text/html，不是可验证的ZIP资产，未当作下载成功。CAN bus的期望文件名由BEVFormer/MapTR官方prepare_dataset.md确认是can_bus.zip，解压期望data/nuscenes/can_bus/；请从nuScenes官方登录下载入口取得，或提供已有本地路径。证据：logs/auxiliary_assets_probe.json。
