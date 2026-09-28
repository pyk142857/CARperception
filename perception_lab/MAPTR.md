# MapTR 车道分隔线与矢量地图教学

模型：MapTR tiny / ResNet-50 / 24 epochs，官方 nuScenes 预训练权重。输入为每帧六路相机，输出为 `divider`（分隔线）、`ped_crossing`（人行横道）、`boundary`（道路边界）的折线及置信度；每条折线 20 个点。它不输出实虚线类别、车道编号或完整道路拓扑。

## 执行

在 `perception_lab` 目录执行：

```bash
# 首次：隔离安装旧版框架，并编译 CUDA 算子。需要 Python 3.10 / CUDA 11.8 编译器。
bash tools/setup_maptr.sh
# 首帧验证；独立保存，不能冒充整场景结果。
envs/maptr/bin/python tools/maptr_mini.py --limit 1 --out outputs/maptr/smoke
# scene-0061 全部 39 个关键帧。
envs/maptr/bin/python tools/maptr_mini.py
# 生成新记录，成功后再替换正在提供的文件。
envs/rerun/bin/python tools/rerun_mini.py --out outputs/rerun/mini_scene.new.rrd --lane-score 0.5
mv outputs/rerun/mini_scene.new.rrd outputs/rerun/mini_scene.rrd
mv outputs/rerun/mini_scene.new.json outputs/rerun/mini_scene.json
envs/rerun/bin/python tools/start_rerun.py
```

Rerun 页面已打开时需要刷新，才会重新读取 `.rrd`。BEV 视图朝上为车辆前方；黄色为分隔线、粉色为人行横道、蓝色为道路边界。三维视图下对应 `ego/maptr`，BEV 下对应 `bev/maptr`。图层每帧清空再写入，不累积旧线。阈值只影响显示；JSON 保留模型解码的全部候选。

## 实现位置

- `tools/maptr_mini.py`：官方模型注册、严格权重加载、原版测试预处理、GPU 推理、逐帧结果和证据摘要。
- `tools/maptr_geometry.py`：考虑相机实际采集时刻的 LiDAR→相机投影、预测折线→车辆坐标转换、结果校验。
- `tools/camera_overlay.py`：将车辆坐标预测折线投影到六路图像，裁剪相机近面和画面边界。采用近似地面高度，不做前景遮挡处理。
- `tools/rerun_mini.py`：校验文件哈希和帧 token，按类别记录 LineStrips3D / LineStrips2D。
- `third_party/MapTR/projects/configs/maptr/maptr_tiny_r50_24e.py`：网络、相机归一化、0.5 倍缩放和 padding 配置。
- `third_party/MapTR/projects/mmdet3d_plugin/maptr/detectors/maptr.py`：图像特征提取和前向推理。
- `third_party/MapTR/projects/mmdet3d_plugin/maptr/modules/transformer.py`：多视角 BEV 特征融合。
- `third_party/MapTR/projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py`：折线点与类别预测。

## 坐标与边界

按官方相机顺序处理：前、右前、左前、后、左后、右后。模型使用 LiDAR 平面坐标；绘图时用标定转换 XY 到现有参考车辆坐标。模型没有高度输出，三维展示的 Z 固定为车辆坐标下 0 米，是近似地面而非预测的三维路面。坡道等情况下不能把三维贴合精度当作模型高度精度。

不加载地图真值进行推理，也不把地图标注画成预测线。mini 没有另行安装 CAN bus 扩展：位置、姿态、航向由位姿元数据提供，其余加速度、角速度、速度字段沿用官方缺失数据的零填充方式。这属于流程教学条件，不宣称复现官方全量精度。

mini 场景可能与预训练训练集重叠；运行和可视化不构成独立评估。需要正式精度评估时，另行准备地图扩展真值及相应数据划分。

旧源码的兼容性调整由 `tools/setup_maptr.sh` 重现；保留真实 CUDA 算子和官方网络，不用模拟预测替代运行。环境不修改 CenterPoint 或 Rerun 的依赖。

上游：[MapTR](https://github.com/hustvl/MapTR)、[nuScenes 地图扩展](https://www.nuscenes.org/tutorials/map_expansion_tutorial.html)。
