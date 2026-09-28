# LiDARSeg：nuScenes mini 逐点语义分割

已完成 scene-0061 的 39 帧、1,354,112 个点的真实 Cylinder3D 推理。输入为当前帧原始 LiDAR XYZ 和反射强度；输出保持原始点顺序的类别 ID 和置信度。地图、三维检测框、相机预测及点云真值不进入网络。采用 nuScenes 专用官方权重，16 个语义类加 0（ignore）。

## 数据与模型

- 官方 mini 标签扩展：[下载](https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2)。404 个关键帧、14,026,208 个逐点标签已全部核对点数；摘要为 `outputs/lidarseg/data_audit.json`。
- 作者模型：[Cylinder3D nuScenes 指南](https://github.com/xinge008/Cylinder3D/blob/master/NUSCENES-GUIDE.md)所链接的 [Google Drive 文件夹](https://drive.google.com/drive/folders/1zSZ9xE4UkKBMCMH0le7KdSxvbyjuuUp8)中的 `model_load_nuscenes.pt`，本地保存为 `checkpoints/cylinder3d_nuscenes.pt`。
- 固定源码 commit：`30a0abb2ca4c657a821a5e9a343934b0789b2365`。
- 类别：barrier、bicycle、bus、car、construction_vehicle、motorcycle、pedestrian、traffic_cone、trailer、truck、driveable_surface、other_flat、sidewalk、terrain、manmade、vegetation。
- 标签扩展的 category.json 增加语义类别；替换前已确认原有类别 token/名称一致，原件保存在 `data/pre_lidarseg`，不影响已有检测框类别解析。

## 复现

从 `perception_lab` 执行。需要已部署的 MapTR 旧框架环境及其编译算子；新机器先执行 `bash tools/setup_maptr.sh`，并准备上述权重。

```bash
bash tools/setup_lidarseg.sh
envs/lidarseg_legacy/bin/python tools/lidarseg_mini.py --limit 1 --out outputs/lidarseg/smoke_check
envs/lidarseg_legacy/bin/python tools/lidarseg_mini.py
envs/rerun/bin/python tools/rerun_mini.py
envs/rerun/bin/python tools/start_rerun.py
```

`envs/lidarseg_legacy` 引用本项目 `envs/maptr` 的包目录，并单独安装相容的 torch-scatter。不要移动其中一个环境后继续使用另一个。没有改动全局 CUDA、现有 CenterPoint 环境或 MapTR 的模型代码。

## 实现

1. `tools/prepare_lidarseg_data.py`：下载、解包、保护已有元数据、按 sample_data token 核对全部标签长度。
2. `tools/lidarseg_utils.py`：与作者一致的柱坐标划分（480×360×32）和九维点特征、32→16 类映射、忽略真值 0 的混淆矩阵。
3. `tools/prepare_cylinder3d.py`：在独立适配目录接入原版 SpConv 1 算子。
4. `tools/lidarseg_mini.py`：官方权重 274 个张量严格加载；点特征网络→体素池化→非对称稀疏卷积→每点类别/概率。完成预测后才读取 GT 用于对比。文件、权重、源码哈希保存在 summary.json。
5. `tools/rerun_mini.py`：验证哈希、帧 token、原始点文件和逐点数量，导出预测着色及独立 GT 视图。

### 为什么保留旧版算子

此官方旧权重使用共享索引缓存的 SpConv 1 行为。直接转 SpConv 2 并为不同卷积核拆分缓存会改变计算：本次首帧实验有效点准确率约 24%，恢复原版算子后为 94.7%。因此最终路径保留原始算子、卷积核布局和缓存语义，只将 checkpoint 的 `polar_` 模块前缀对应到当前 `cylinder_` 名称；未采用部分权重加载。失败实验保留在本地 smoke 目录供排障，不供查看器加载。

## Rerun 查看

- 左侧三维区域的 **LiDARSeg prediction / detections**：逐点预测颜色，叠加既有检测框、轨迹与地图线；`ego/lidar`。
- 同一区域切换 **LiDARSeg ground truth** 页签：只显示官方逐点标注；`lidarseg_gt/points`。
- BEV 点云使用预测类别颜色，MapTR 线和六路相机叠加仍保留。
- 道路为青绿色、植被为绿色、人造结构为浅黄色、汽车为橙色；点上记录 class_ids，置信度在预测实体属性中。完整配色见 `lidarseg_utils.PALETTE`。
- 39 帧在同一时间轴更新；点云分割不会自动修改 CenterPoint 框或跟踪结果。

## 验证与指标边界

预处理在首帧全部 34,688 个点上与作者实现逐项完全一致。全部结果与原始点数、顺序对齐。全套 25 项测试通过；RRD 的预测和 GT 类别逐点与输出 NPZ 一致。

当前 scene-0061 诊断：有效点准确率 **95.16%**，非零并集类别平均 IoU **71.99%**；详细每类 IoU/混淆矩阵在 `outputs/lidarseg/summary.json`。mini 场景与预训练数据存在重叠，这些数字用于检查流程和查看错误，不是独立验证集成绩，也不代表复现作者全量基准。

完整 mini 标签已补齐，连续模型推理范围仍是当前教学场景的 39 帧；未声称全部 404 帧已推理。数据、权重、NPZ 与 RRD 不提交 Git。
